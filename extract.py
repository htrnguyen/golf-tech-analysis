import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

# Thêm đường dẫn src để import model
sys.path.insert(0, os.path.abspath('src'))

from model import EventDetector
from eval import ToTensor, Normalize
from tqdm import tqdm

import argparse
import json

def smooth_probs(probs, window_size=5):
    """Làm mượt xác suất bằng Moving Average"""
    smoothed = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        smoothed[:, i] = np.convolve(probs[:, i], np.ones(window_size)/window_size, mode='same')
    # Chia lại để tổng xác suất mỗi frame = 1
    return smoothed / smoothed.sum(axis=1, keepdims=True)

def run_ai_extraction(video_path, slow_factor=1.0, output_dir=None):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file {video_path}")
        return

    video_name = os.path.basename(video_path)
    video_prefix = os.path.splitext(video_name)[0]
    
    # Nếu không có output_dir, dùng mặc định
    if output_dir is None:
        output_dir = os.path.join('output', video_prefix)
        
    model_path = 'models/swingnet_1800.pth.tar'
    phases_dir = os.path.join(output_dir, 'phases')
    
    # Thêm số thứ tự để đảm bảo sắp xếp đúng trong thư mục
    labels = ['1_Address', '2_Toe-up', '3_Mid-Backswing', '4_Top', 
              '5_Mid-Downswing', '6_Impact', '7_Mid-Follow-Through', '8_Finish']
    for label in labels: 
        os.makedirs(os.path.join(phases_dir, label), exist_ok=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EventDetector(pretrain=False, width_mult=1., lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)
    
    save_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device).eval()
    print(f"Mô hình đã sẵn sàng trên {device} (Slow factor: {slow_factor})")

    transform = transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    print(f"Đang đọc video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, img = cap.read()
        if not ret: break
        raw_frames.append(img)
    cap.release()

    if not raw_frames:
        print("Không tìm thấy frame nào.")
        return

    # Frame Interpolation (Nội suy tuyến tính)
    if slow_factor < 1.0:
        print(f"Đang nội suy frame (Slow-mo {slow_factor}x)...")
        steps = int(1.0 / slow_factor)
        interpolated_full_res = []
        for i in range(len(raw_frames) - 1):
            f1 = raw_frames[i]
            f2 = raw_frames[i+1]
            for s in range(steps):
                alpha = s / steps
                mixed = cv2.addWeighted(f1, 1-alpha, f2, alpha, 0)
                interpolated_full_res.append(mixed)
        interpolated_full_res.append(raw_frames[-1])
        full_res_frames = interpolated_full_res
        
        # Lưu file video slow-motion vật lý
        slow_video_path = os.path.join(output_dir, "slow_motion.mp4")
        print(f"Đang ghi file slow motion: {slow_video_path}")
        
        # Lấy lại FPS gốc (hoặc mặc định 30)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0: fps = 30
        
        h, w = full_res_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(slow_video_path, fourcc, fps, (w, h))
        
        for frame in full_res_frames:
            out.write(frame)
        out.release()
        print(f"Đã tạo video slow motion: {slow_video_path}")
        
    else:
        full_res_frames = raw_frames

    # Tiền xử lý cho AI
    print("Đang tiền xử lý cho AI...")
    images = []
    input_size = 160
    for img in tqdm(full_res_frames):
        h, w = img.shape[:2]
        ratio = input_size / max(h, w)
        new_size = (int(w * ratio), int(h * ratio))
        resized = cv2.resize(img, new_size)
        delta_w, delta_h = input_size - new_size[0], input_size - new_size[1]
        top, bottom, left, right = delta_h // 2, delta_h - (delta_h // 2), delta_w // 2, delta_w - (delta_w // 2)
        b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0.406*255, 0.456*255, 0.485*255])
        images.append(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
    
    sample = transform({'images': np.asarray(images), 'labels': np.zeros(len(images))})
    img_tensor = sample['images'].unsqueeze(0).to(device)
    
    print("Đang chạy AI Inference...")
    with torch.no_grad():
        seq_length, batch, all_logits = 48, 0, []
        while batch * seq_length < img_tensor.shape[1]:
            end_idx = min((batch + 1) * seq_length, img_tensor.shape[1])
            all_logits.append(model(img_tensor[:, batch * seq_length:end_idx, :, :, :]).cpu().numpy())
            batch += 1
        
        logits_concat = np.concatenate(all_logits, axis=0)
        probs = F.softmax(torch.tensor(logits_concat), dim=1).numpy()
        
        # Làm mượt kết quả để giảm nhiễu (đặc biệt cho DTL)
        probs = smooth_probs(probs, window_size=5)
        
        # Áp dụng thuật toán Anchor-based Bidirectional Search
        # 1. Tìm sự kiện "Neo" (Anchor) đáng tin cậy nhất (thường là Impact hoặc Top)
        # Chỉ xét 8 lớp sự kiện đầu tiên, bỏ qua lớp số 9 (No Event/Background)
        probs_events = probs[:, :8]
        max_probs = np.max(probs_events, axis=0) # Max prob của từng class sự kiện
        anchor_class = np.argmax(max_probs) # Class có độ tự tin cao nhất trong 8 sự kiện
        anchor_frame = np.argmax(probs_events[:, anchor_class])
        
        print(f"Detected Anchor: {labels[anchor_class]} at frame {anchor_frame} (conf: {max_probs[anchor_class]:.2f})")
        
        events = np.zeros(8, dtype=int)
        events[anchor_class] = anchor_frame
        
        # 2. Đi lùi: Tìm các pha trước Anchor
        current_limit = anchor_frame
        for i in range(anchor_class - 1, -1, -1):
            if current_limit > 0:
                # Tìm max trong vùng cho phép [0, current_limit]
                segment = probs[0:current_limit, i]
                if len(segment) > 0:
                    events[i] = np.argmax(segment)
                else:
                    events[i] = 0
            else:
                events[i] = 0
            current_limit = events[i]
            
        # 3. Đi tiến: Tìm các pha sau Anchor
        current_limit = anchor_frame
        total_frames = probs.shape[0]
        for i in range(anchor_class + 1, 8):
            if current_limit < total_frames - 1:
                # Tìm max trong vùng cho phép [current_limit + 1, end]
                # Bắt buộc tiến ít nhất 1 frame
                segment = probs[current_limit + 1:, i]
                if len(segment) > 0:
                    events[i] = (current_limit + 1) + np.argmax(segment)
                else:
                    events[i] = total_frames - 1
            else:
                events[i] = total_frames - 1
            current_limit = events[i]
            
        print(f"Detected Events (Frames): {events}")

        # Lưu thông tin frame index để visual_report sử dụng
        event_metadata = {}
        for i, frame_idx in enumerate(events):
            if frame_idx < len(full_res_frames):
                cv2.imwrite(os.path.join(phases_dir, labels[i], f"{video_prefix}.jpg"), full_res_frames[frame_idx])
                event_metadata[labels[i]] = int(frame_idx)
        
        # Lấy FPS để đồng bộ phía frontend
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0: fps = 30

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({"event_frames": event_metadata, "slow_factor": slow_factor, "fps": fps}, f)
    
    print(f"Xong! Ảnh trích xuất lưu tại {phases_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default='data/output_video_-M5SITXMA2Y.mp4')
    parser.add_argument('--slow', type=float, default=1.0, help='Tỉ lệ làm chậm video (0.5, 0.2, ...)')
    parser.add_argument('--output_dir', type=str, default=None, help='Đường dẫn thư mục đầu ra')
    args = parser.parse_args()
    
    run_ai_extraction(args.video_path, args.slow, args.output_dir)
