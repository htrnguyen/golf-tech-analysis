import cv2
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import argparse

# Cấu hình encoding cho stdout để in tiếng Việt
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Tắt logs từ TensorFlow/MediaPipe trong production
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt TF logs
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Vai - Khuỷu - Cổ tay
    (11, 23), (12, 24), (23, 24), # Vai - Hông - Hông
    (23, 25), (24, 26), (25, 27), (26, 28), # Hông - Gối - Cổ chân
]

def draw_vietnamese_text(img_cv, text, position, font_size=24, color=(255, 255, 255), max_width=None):
    """Vẽ tiếng Việt lên ảnh OpenCV sử dụng Pillow"""
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Hardcode font path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "font", "ARIAL.TTF")
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    if max_width:
        lines = textwrap.wrap(text, width=max_width)
        y = position[1]
        for line in lines:
            draw.text((position[0], y), line, font=font, fill=color)
            y += font_size + 5
    else:
        draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_glass_panel(img, pt1, pt2, color=(0, 0, 0), alpha=0.5):
    """Vẽ bảng điều khiển bán trong suốt (Glassmorphism effect)"""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def reengineer_video(json_path, video_path, output_path=None, production=False):
    """
    Áp dụng dữ liệu JSON lên video gốc (nguyên kích thước) với overlay cao cấp.
    """
    if not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy file JSON tại {json_path}")
        return
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        return

    # 1. Đọc Master JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    analysis = data.get("analysis", {})
    event_frames = metadata.get("event_frames", {})
    fps_meta = metadata.get("fps", 30)

    # 2. Mở video gốc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0: fps_video = fps_meta
    
    # Xử lý ghi đè (Overwrite logic)
    is_overwrite = False
    final_output_path = output_path
    
    if output_path is None:
        # Mặc định ghi đè (overwrite) lên chính video gốc
        final_output_path = video_path
    
    # Nếu đường dẫn output trùng với video gốc, cần dùng file tạm
    if os.path.abspath(final_output_path) == os.path.abspath(video_path):
        is_overwrite = True
        temp_output_path = final_output_path + ".tmp.mp4"
    else:
        temp_output_path = final_output_path

    if not production:
        print(f"--- Đang tạo video Premium: {os.path.basename(video_path)} ({vw}x{vh}) ---")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps_video, (vw, vh))

    idx_to_phase = {int(v): k for k, v in event_frames.items()}
    current_phase = "1_Address"
    
    # Scaling: giữ nguyên ban đầu cho video bình thường, chỉ giảm cho video cực nhỏ
    baseline_height = 720
    scale = vh / baseline_height
    
    # Font sizes
    if vh < 240:
        # Video cực nhỏ - giảm mạnh
        base_font_size = max(6, int(10 * scale))
        small_font_size = max(5, int(8 * scale))
        title_font_size = max(7, int(12 * scale))
    else:
        # Video bình thường - giữ nguyên kích thước lớn
        base_font_size = max(18, int(vh / 25))
        small_font_size = int(base_font_size * 0.7)
        title_font_size = int(base_font_size * 1.2)
    
    # Layout
    line_spacing = int(base_font_size * 1.8)
    margin = int(vw * 0.02)
    
    # Skeleton
    skeleton_line_thickness = max(1, int(2 * scale))
    skeleton_joint_radius = max(1, int(3 * scale))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. KIỂM TRA EVENT ĐỂ TẠO CÚ KHỰNG (FREEZE-FRAME)
        if frame_idx in idx_to_phase:
            current_phase = idx_to_phase[frame_idx]

            # Tạo frame đặc biệt có overlay cao cấp
            pause_frame = frame.copy()
            
            # --- VẼ SKELETON (MỎNG & TINH TẾ) ---
            phase_data = analysis.get("phases", {}).get(current_phase)
            if phase_data and "raw_landmarks" in phase_data:
                landmarks = phase_data["raw_landmarks"]
                # Bones (Line thickness: 2)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        l1 = landmarks[start_idx]
                        l2 = landmarks[end_idx]
                        if l1["visibility"] > 0.5 and l2["visibility"] > 0.5:
                            p1 = (int(l1["x"] * vw), int(l1["y"] * vh))
                            p2 = (int(l2["x"] * vw), int(l2["y"] * vh))
                            cv2.line(pause_frame, p1, p2, (0, 255, 157), skeleton_line_thickness)
                
                # Joints (Ultra-sleek dots)
                # 1. Xử lý khuôn mặt: 1 chấm mũi + 1 viền tròn đầu
                nose = landmarks[0]
                ear_l = landmarks[7]
                ear_r = landmarks[8]
                if nose["visibility"] > 0.5:
                    p_nose = (int(nose["x"] * vw), int(nose["y"] * vh))
                    # Vẽ chấm mũi
                    cv2.circle(pause_frame, p_nose, 2, (255, 255, 255), -1)
                    
                    # Vẽ viền tròn đầu (Face outline)
                    if ear_l["visibility"] > 0.5 and ear_r["visibility"] > 0.5:
                        p_l = np.array([ear_l["x"] * vw, ear_l["y"] * vh])
                        p_r = np.array([ear_r["x"] * vw, ear_r["y"] * vh])
                        radius = int(np.linalg.norm(p_l - p_r) * 0.8)
                        cv2.circle(pause_frame, p_nose, radius, (0, 255, 157), 1)

                # 2. Xử lý các khớp còn lại (Body)
                for i, lm in enumerate(landmarks):
                    if lm["visibility"] > 0.5:
                        # Bỏ qua các điểm mặt 0-10 vì đã xử lý riêng
                        if i <= 10:
                            continue
                        
                        p = (int(lm["x"] * vw), int(lm["y"] * vh))
                        # Các khớp thân mình scale theo video size
                        cv2.circle(pause_frame, p, skeleton_joint_radius, (255, 255, 255), -1)
                        cv2.circle(pause_frame, p, skeleton_joint_radius + 1, (0, 255, 157), 1)

            # Panel size
            if vh < 240:
                # Video nhỏ - panel lớn hơn
                panel_w = int(vw * 0.85)
                panel_h = int(vh * 0.6)
                text_padding = 5
                panel_margin = 3
            else:
                # Video bình thường - giữ nguyên
                panel_w = int(vw * 0.45)
                panel_h = int(vh * 0.35)
                text_padding = 20
                panel_margin = 10
            
            # Vẽ panel mờ ở góc trên trái
            pause_frame = draw_glass_panel(pause_frame, (panel_margin, panel_margin), 
                                         (panel_margin + panel_w, panel_margin + panel_h), alpha=0.6)

            phase_display_parts = current_phase.split('_', 1)
            phase_display_name = phase_display_parts[1].upper() if len(phase_display_parts) > 1 else current_phase.upper()
            
            # Text nội dung - sử dụng text_padding thay vì hardcode
            text_x = panel_margin + text_padding
            text_y = panel_margin + int(title_font_size * 1.5)
            
            # Title
            pause_frame = draw_vietnamese_text(pause_frame, f"GIAI ĐOẠN: {phase_display_name}", 
                                            (text_x, text_y), title_font_size, (0, 255, 157))
            
            if phase_data:
                score = phase_data.get("score", 0)
                comments = " | ".join(phase_data.get("comments", []))
                # Điểm số
                pause_frame = draw_vietnamese_text(pause_frame, f"ĐIỂM: {score}/10", 
                                                (text_x, text_y + line_spacing), base_font_size, (255, 255, 255))
                # Nhận xét
                pause_frame = draw_vietnamese_text(pause_frame, f"NHẬN XÉT: {comments}", 
                                                (text_x, text_y + line_spacing * 2), small_font_size, (200, 200, 200), max_width=panel_w - 40)
            # Freeze for 2 seconds
            for _ in range(int(fps_video * 2.0)):
                out.write(pause_frame)
        
        # 4. GHI FRAME VIDEO BÌNH THƯỜNG
        display_frame = frame.copy()
        phase_display_parts = current_phase.split('_', 1)
        phase_display_name = phase_display_parts[1].upper() if len(phase_display_parts) > 1 else current_phase.upper()
        
        # Nhãn nhỏ góc trên
        label_w = int(vw * 0.3)
        display_frame = draw_glass_panel(display_frame, (20, 20), (20 + label_w, 20 + line_spacing), alpha=0.4)
        display_frame = draw_vietnamese_text(display_frame, f"PHA: {phase_display_name}", 
                                           (30, 25), small_font_size, (0, 255, 157))
        
        out.write(display_frame)
        frame_idx += 1

    cap.release()
    out.release()

    if is_overwrite:
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rename(temp_output_path, final_output_path)

    if not production:
        print(f"--- HOÀN TẤT! Video lưu tại: {final_output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf Video Re-engineer Tool")
    parser.add_argument("--json", required=True, help="Đường dẫn file master_data.json")
    parser.add_argument("--video", required=True, help="Đường dẫn video gốc .mp4")
    parser.add_argument("--output", help="Đường dẫn file đầu ra")
    
    args = parser.parse_args()
    reengineer_video(args.json, args.video, args.output)
