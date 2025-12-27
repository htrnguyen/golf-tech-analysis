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
        
        # Tính chiều cao thân người (Torso Height) để chuẩn hóa
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        torso_height = abs(mid_shoulder_y - mid_hip_y)
        
        if torso_height == 0: return "Unknown"
        
        # Tỷ lệ chiều rộng vai / chiều cao thân
        ratio = shoulder_width / torso_height
        
        # Heuristic: Nếu tỷ lệ > 0.5 (vai rộng bằng nửa lưng) -> Face-on
        if ratio > 0.4: # Hạ threshold xuống chút cho an toàn
            return "Face-on (Trực diện)"
        else:
            return "Down-the-Line (Dọc)"


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
        
        # Cấu hình phong cách vẽ tinh tế (mỏng hơn, nhỏ hơn)
        landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1) # Neon Green thon gọn
        connection_style = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1) # Đường nối trắng mỏng
        
        # Vẽ khung xương
        self.mp_drawing.draw_landmarks(
            annotated_image, landmarks_proto, self.mp_pose.POSE_CONNECTIONS,
            landmark_style,
            connection_style
        )
        
        return annotated_image

    def analyze_phase(self, phase_name, landmarks):
        if not landmarks:
            return {"score": 0, "comments": ["Không tìm thấy tư thế"], "data": {}}

        analysis = {"score": 10.0, "comments": [], "data": {}, "raw_landmarks": []}
        
        # Trích xuất tọa độ thô của toàn bộ khớp xương để kỹ sư khác sử dụng
        for i, lm in enumerate(landmarks):
            analysis["raw_landmarks"].append({
                "id": i,
                "name": self.mp_pose.PoseLandmark(i).name,
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 4)
            })

        # Lấy các điểm mốc quan trọng để tính toán logic nội bộ
        l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        l_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        r_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        l_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        r_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Phân tích cụ thể từng pha (Thang điểm 10)
        # Sử dụng endswith để khớp với tên có hoặc không có số thứ tự
        if phase_name.endswith('Address'):
            # 1. Độ rộng chân (so với vai)
            shoulder_width = np.abs(l_shoulder[0] - r_shoulder[0])
            stance_width = np.abs(l_ankle[0] - r_ankle[0])
            ratio = stance_width / shoulder_width if shoulder_width > 0 else 0
            analysis["data"]["stance_ratio"] = ratio
            if ratio < 0.8:
                analysis["score"] -= 1.0
                analysis["comments"].append("Tư thế đứng quá hẹp")
            elif ratio > 1.4:
                analysis["score"] -= 1.0
                analysis["comments"].append("Tư thế đứng quá rộng")

        elif phase_name.endswith('Top'):
            # 2. Độ thẳng tay trái (Lead arm - giả sử golfer thuận tay phải)
            lead_arm_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            analysis["data"]["lead_arm_angle"] = lead_arm_angle
            if lead_arm_angle < 150:
                analysis["score"] -= 2.0
                analysis["comments"].append("Tay trái bị cong quá nhiều (Chicken Wing)")
            
            # 3. Góc xoay vai (đơn giản hóa)
            shoulder_tilt = np.abs(l_shoulder[1] - r_shoulder[1])
            analysis["data"]["shoulder_tilt"] = shoulder_tilt

        elif phase_name.endswith('Impact'):
            # 4. Độ thẳng tay trái lúc Impact
            impact_arm_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            if impact_arm_angle < 160:
                analysis["score"] -= 2.5
                analysis["comments"].append("Tay trái không thẳng, mất lực")
            
            # 5. Hip rotation (Lead hip vs Back hip)
            hip_openness = l_hip[0] - r_hip[0]
            analysis["data"]["hip_openness"] = hip_openness
            
        elif phase_name.endswith('Finish'):
            # 6. Thăng bằng (Trọng tâm dồn về chân trái)
            # Kiểm tra thăng bằng đơn giản bằng vị trí hông so với chân lead
            if l_hip[0] > l_ankle[0] + 0.1 or l_hip[0] < l_ankle[0] - 0.1:
                analysis["score"] -= 1.5
                analysis["comments"].append("Kết thúc thiếu thăng bằng")

        if not analysis["comments"]:
            analysis["comments"].append("Đạt chuẩn")

        analysis["score"] = round(max(0, analysis["score"]), 1)
        return analysis

    def process_video_results(self, output_dir):
        # Lấy video_id từ tên thư mục output
        video_id = os.path.basename(output_dir)
        extraction_dir = os.path.join(output_dir, 'phases')
        
        report = {"video_id": video_id, "phases": {}, "overall_score": 0, "view_angle": "Unknown"}
        
        total_score = 0
        valid_phases = 0

        # Mảng lưu landmarks Address để detect view
        address_landmarks = None

        for label in self.labels:
            img_path = os.path.join(extraction_dir, label, f"{video_id}.jpg")
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            landmarks_list, _ = self.get_landmarks(image)
            
            # Lưu landmarks của Address
            if label == '1_Address' and landmarks_list:
                address_landmarks = landmarks_list

            phase_analysis = self.analyze_phase(label, landmarks_list)

            report["phases"][label] = phase_analysis
            total_score += phase_analysis["score"]
            valid_phases += 1

        if valid_phases > 0:
            report["overall_score"] = round(total_score / valid_phases, 1)

        # Detect View Angle một lần duy nhất
        if address_landmarks:
            report["view_angle"] = self.detect_view_angle(address_landmarks)
        
        # Lưu báo cáo vào thư mục output của video
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
