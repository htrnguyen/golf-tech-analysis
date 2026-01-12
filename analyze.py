import sys
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import mediapipe as mp
import numpy as np
import os
import json

class GolfDiagnosticEngine:
    """
    Hệ thống chẩn đoán kỹ thuật Golf dựa trên tư thế (Pose Estimation).
    Sử dụng MediaPipe để trích xuất điểm mốc và heuristics hình học để chấm điểm.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.labels = ['1_Address', '2_Toe-up', '3_Mid-Backswing', '4_Top', 
                       '5_Mid-Downswing', '6_Impact', '7_Mid-Follow-Through', '8_Finish']

    def detect_view_angle(self, landmarks):
        """
        Xác định góc quay (Face-on hay Down-the-line) dựa trên độ rộng vai ở Address.
        Face-on: Hai vai cách xa nhau theo trục X.
        DTL: Hai vai gần nhau (che khuất).
        """
        if not landmarks: return "Unknown"
        
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        torso_height = abs(mid_shoulder_y - mid_hip_y)
        
        if torso_height == 0: return "Unknown"
        
        ratio = shoulder_width / torso_height
        
        # Nếu tỷ lệ > 0.35 -> Face-on.
        if ratio > 0.35: 
            return "Face-on"
        else:
            return "Down-the-Line"

    def calculate_angle(self, a, b, c):
        """Tính góc (độ) giữa 3 điểm a, b, c (b là đỉnh)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def get_landmarks(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None
        return results.pose_landmarks.landmark, results.pose_landmarks

    def draw_annotations(self, image, landmarks_proto, phase_name, analysis):
        """Vẽ landmarks tinh tế và chuyên nghiệp hơn"""
        annotated_image = image.copy()
        
        landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        connection_style = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        
        self.mp_drawing.draw_landmarks(
            annotated_image, landmarks_proto, self.mp_pose.POSE_CONNECTIONS,
            landmark_style,
            connection_style
        )
        return annotated_image

    def analyze_phase(self, phase_name, landmarks, address_landmarks=None, view_angle="Unknown"):
        if not landmarks:
            return {"score": 0, "comments": ["Không tìm thấy tư thế"], "data": {}}

        # Điểm tối đa mặc định là 10.0
        analysis = {"score": 10.0, "comments": [], "data": {}, "raw_landmarks": []}
        
        # Helper lấy tọa độ (x, y)
        def get_xy(idx):
            return [landmarks[idx].x, landmarks[idx].y]

        l_shoulder = get_xy(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = get_xy(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_elbow = get_xy(self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        r_elbow = get_xy(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        l_wrist = get_xy(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        r_wrist = get_xy(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        l_hip = get_xy(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip = get_xy(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        l_knee = get_xy(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        r_knee = get_xy(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        l_ankle = get_xy(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        r_ankle = get_xy(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        nose = get_xy(self.mp_pose.PoseLandmark.NOSE.value)
        ear_l = get_xy(self.mp_pose.PoseLandmark.LEFT_EAR.value)
        ear_r = get_xy(self.mp_pose.PoseLandmark.RIGHT_EAR.value)

        # Trích xuất landmarks thô
        for i, lm in enumerate(landmarks):
            analysis["raw_landmarks"].append({
                "id": i, "name": self.mp_pose.PoseLandmark(i).name,
                "x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4), "visibility": round(lm.visibility, 4)
            })

        # --- CHUẨN HÓA TỌA ĐỘ (Normalization) ---
        # Gốc tọa độ tương đối: Trung điểm 2 mắt cá chân
        current_ankle_center_x = (l_ankle[0] + r_ankle[0]) / 2
        current_ankle_center_y = (l_ankle[1] + r_ankle[1]) / 2 # Dùng Y để check Head Dip
        
        addr_ankle_center_x = 0
        addr_ankle_center_y = 0
        addr_nose_x_rel = 0
        addr_nose_y_rel = 0
        addr_hip_x_rel = 0
        
        if address_landmarks:
            al_ankle_l = address_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            al_ankle_r = address_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            addr_ankle_center_x = (al_ankle_l.x + al_ankle_r.x) / 2
            addr_ankle_center_y = (al_ankle_l.y + al_ankle_r.y) / 2
            
            al_nose = address_landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            al_hip = (address_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x + 
                      address_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
            
            addr_nose_x_rel = al_nose.x - addr_ankle_center_x
            addr_nose_y_rel = al_nose.y - addr_ankle_center_y # Y relative (height)
            addr_hip_x_rel = al_hip - addr_ankle_center_x

        # Tọa độ tương đối hiện tại
        curr_nose_x_rel = nose[0] - current_ankle_center_x
        curr_nose_y_rel = nose[1] - current_ankle_center_y
        curr_hip_x_rel = ((l_hip[0] + r_hip[0])/2) - current_ankle_center_x
        
        hip_center_x = (l_hip[0] + r_hip[0]) / 2

        # --- LOGIC PHÂN TÍCH CHUYÊN SÂU (HEAD-TO-TOE) ---

        # 1. ADDRESS (Chuẩn bị)
        if phase_name.endswith('Address'):
            # A. Stance Width (Độ rộng chân) - Face-on only
            if view_angle == "Face-on":
                shoulder_width = np.abs(l_shoulder[0] - r_shoulder[0])
                stance_width = np.abs(l_ankle[0] - r_ankle[0])
                if shoulder_width > 0:
                    ratio = stance_width / shoulder_width
                    analysis["data"]["stance_ratio"] = ratio
                    # Stricter thresholds
                    if ratio < 0.95: 
                        analysis["score"] -= 1.0
                        analysis["comments"].append("Tư thế đứng quá hẹp (Narrow Stance) - Mất thăng bằng")
                    elif ratio > 1.6:
                        analysis["score"] -= 1.0
                        analysis["comments"].append("Tư thế đứng quá rộng (Wide Stance) - Khó xoay người")
            
            # B. Knee Flex (Độ khuỵu gối)
            avg_knee_angle = (self.calculate_angle(l_hip, l_knee, l_ankle) + 
                              self.calculate_angle(r_hip, r_knee, r_ankle)) / 2
            analysis["data"]["knee_angle"] = avg_knee_angle
            if avg_knee_angle > 175:
                analysis["score"] -= 1.5 # Phạt nặng hơn
                analysis["comments"].append("Đầu gối khóa thẳng (Locking Knees) - Cần chùng gối")
            elif avg_knee_angle < 130: # Stricter
                analysis["score"] -= 1.0
                analysis["comments"].append("Đầu gối khuỵu quá thấp (Sitting too much)")

            # C. Spine Angle (Trục cột sống) - Face-on
            if view_angle == "Face-on":
                # Vai trái cần cao hơn vai phải (với người thuận tay phải)
                # Y càng nhỏ càng cao
                if l_shoulder[1] > r_shoulder[1]: 
                    analysis["score"] -= 0.5
                    analysis["comments"].append("Vai trái thấp hơn vai phải (Nên setup vai phải thấp hơn)")

        # 2. TOP OF BACKSWING (Đỉnh backswing)
        elif phase_name.endswith('Top'):
            # A. Chicken Wing (Tay trái gập)
            lead_arm_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            analysis["data"]["lead_arm_angle"] = lead_arm_angle
            threshold = 150 if view_angle == "Face-on" else 135 # Stricter threshold
            if lead_arm_angle < threshold:
                analysis["score"] -= 2.0
                analysis["comments"].append("Tay trái bị gập (Chicken Wing) - Mất bán kính swing")

            # B. Flying Elbow (Khuỷu tay phải nâng quá cao)
            # Góc nách phải (Vai phải - Khuỷu phải - Thân mình)
            # Ước lượng trục thân mình bằng Hip
            r_armpit_angle = self.calculate_angle(r_hip, r_shoulder, r_elbow)
            if r_armpit_angle > 100: # Nếu khuỷu tay giơ cao quá vai
                 analysis["score"] -= 1.0
                 analysis["comments"].append("Khuỷu tay phải bay quá cao (Flying Elbow)")

            # C. Head Sway (Đầu lắc ngang) - Face-on
            if address_landmarks and view_angle == "Face-on":
                sway = curr_nose_x_rel - addr_nose_x_rel
                analysis["data"]["head_sway"] = sway
                if sway > 0.05: # Stricter > 5% movement right
                    analysis["score"] -= 1.5
                    analysis["comments"].append("Đầu di chuyển lắc sang phải (Sway)")
                elif sway < -0.05: # Reverse Sway
                    analysis["score"] -= 1.5
                    analysis["comments"].append("Đầu ngả ngược về mục tiêu (Reverse Pivot)")

            # D. Hip Slide / Reverse Pivot
            if view_angle == "Face-on":
                 if nose[0] < hip_center_x - 0.05: # Reverse Spine
                    analysis["score"] -= 2.5
                    analysis["comments"].append("Trục cột sống nghiêng ngược (Reverse Spine Angle) - Nguy hiểm")

            # E. Right Leg Straightening (Chân phải duỗi thẳng đuột)
            r_leg_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
            if r_leg_angle > 175:
                analysis["score"] -= 1.0
                analysis["comments"].append("Chân phải duỗi quá thẳng - Mất tư thế (Loss of Posture)")

        # 3. MID-DOWNSWING (Giữa downswing - kiểm tra Casting)
        elif phase_name.endswith('Mid-Downswing'):
            # Casting (Nhả cổ tay quá sớm)
            # Góc cổ tay (Khuỷu - Cổ tay - Ngón tay?? MediaPipe không có ngón tay rõ)
            # Dùng tạm Góc Cổ tay - Khuỷu tay - Vai (Lag angle) không chính xác lắm với 3 điểm.
            # Thay vào đó check góc cánh tay phải.
            pass

        # 4. IMPACT (Tiếp bóng)
        elif phase_name.endswith('Impact'):
            # A. Weight Shift (Chuyển trọng tâm) - Quan trọng nhất
            if address_landmarks and view_angle == "Face-on":
                shift = curr_hip_x_rel - addr_hip_x_rel 
                analysis["data"]["weight_shift"] = shift
                
                if shift > -0.02: # Hông chưa qua vị trí Address (chưa sang trái)
                    analysis["score"] -= 3.0 # Phạt nặng
                    analysis["comments"].append("Thiếu chuyển trọng tâm sang trái (Hanging Back)")
            
            # B. Scooping / Chicken Wing at Impact
            impact_lead_arm_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            if impact_lead_arm_angle < 160:
                analysis["score"] -= 2.0
                analysis["comments"].append("Tay trái gập khi tiếp bóng (Scooping) - Mất lực")
            
            # C. Head Position (Đầu phải ở sau bóng)
            if hip_center_x > nose[0] + 0.05: # Hông sang trái nhiều hơn đầu -> Tốt
                pass
            elif nose[0] < hip_center_x: # Đầu lao theo hông
                analysis["score"] -= 1.5
                analysis["comments"].append("Đầu lao về trước quá sớm (Lunging)")

            # D. Head Level (Đầu không được thụt xuống quá thấp - Head Dip)
            if address_landmarks:
                dip = curr_nose_y_rel - addr_nose_y_rel # Positive = Lower (Y grows down)
                if dip > 0.08: # Thấp xuống quá nhiều
                    analysis["score"] -= 1.0
                    analysis["comments"].append("Đầu nhún xuống quá thấp (Head Dip)")
                elif dip < -0.05: # Nhổm lên
                    analysis["score"] -= 1.0
                    analysis["comments"].append("Nhổm người lên sớm (Early Extension)")

        # 5. FOLLOW THROUGH / FINISH
        elif phase_name.endswith('Finish'):
            # A. Balance (Thăng bằng)
            if view_angle == "Face-on":
                # Trục hông phải nằm trọn trên chân trái
                hip_vs_ankle = l_hip[0] - l_ankle[0]
                if abs(hip_vs_ankle) > 0.1: 
                    analysis["score"] -= 1.5
                    analysis["comments"].append("Kết thúc chưa thăng bằng trên chân trái")
            
            # B. Pose Finish
            # Hai tay nên cao qua vai trái
            hands_y = (l_wrist[1] + r_wrist[1]) / 2
            if hands_y > l_shoulder[1]: # Tay thấp hơn vai (Y lớn hơn)
                 analysis["score"] -= 1.0
                 analysis["comments"].append("Kết thúc tay quá thấp (Low Finish)")

        # Fallback comments
        if not analysis["comments"]:
            analysis["comments"].append("Tư thế tốt, kỹ thuật ổn định")

        # Clamp score & format
        analysis["score"] = round(max(0, min(10, analysis["score"])), 1)
        return analysis

    def process_video_results(self, output_dir, return_dict=False):
        video_id = os.path.basename(output_dir)
        extraction_dir = os.path.join(output_dir, 'phases')
        
        report = {"video_id": video_id, "phases": {}, "overall_score": 0, "view_angle": "Unknown"}
        
        total_score = 0
        valid_phases = 0

        address_landmarks = None
        address_path = os.path.join(extraction_dir, '1_Address', f"{video_id}.jpg")
        
        view_angle = "Unknown"
        if os.path.exists(address_path):
             img = cv2.imread(address_path)
             lm_list, _ = self.get_landmarks(img)
             if lm_list:
                 address_landmarks = lm_list
                 view_angle = self.detect_view_angle(address_landmarks)
        
        report["view_angle"] = view_angle

        for label in self.labels:
            img_path = os.path.join(extraction_dir, label, f"{video_id}.jpg")
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            landmarks_list, _ = self.get_landmarks(image)
            
            phase_analysis = self.analyze_phase(label, landmarks_list, address_landmarks, view_angle)

            report["phases"][label] = phase_analysis
            total_score += phase_analysis["score"]
            valid_phases += 1

        if valid_phases > 0:
            report["overall_score"] = round(total_score / valid_phases, 1)

        if return_dict:
            return report
        else:
            report_path = os.path.join(output_dir, 'report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            return report

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Thư mục output của video (ví dụ: output/video_01)')
    args = parser.parse_args()
    
    engine = GolfDiagnosticEngine()
    result = engine.process_video_results(args.output_dir)
    print(json.dumps(result, indent=4, ensure_ascii=False))
