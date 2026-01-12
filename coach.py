import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import os

class GolfCoachEngine:
    """
    Công cụ Huấn luyện Golf: Chấm điểm và gợi ý bài tập.
    - Áp dụng trọng số cho từng pha (Impact quan trọng nhất).
    - So sánh lỗi với 'Drill Library' để đưa ra bài tập khắc phục.
    """
    def __init__(self):
        # Trọng số cho từng pha (Impact là quan trọng nhất)
        self.weights = {
            '1_Address': 0.1,
            '2_Toe-up': 0.05,
            '3_Mid-Backswing': 0.05,
            '4_Top': 0.2,
            '5_Mid-Downswing': 0.1,
            '6_Impact': 0.35,
            '7_Mid-Follow-Through': 0.05,
            '8_Finish': 0.1
        }
        
        # Thư viện bài tập gợi ý
        self.drill_library = {
            # Stance & Address
            "Tư thế đứng quá rộng (Wide Stance)": "Thu hẹp khoảng cách chân bằng chiều rộng vai để xoay hông dễ dàng hơn.",
            "Tư thế đứng quá hẹp (Narrow Stance)": "Mở rộng chân thêm chút (bằng vai) để tạo nền tảng vững chắc.",
            "Đầu gối quá thẳng, thiếu sự linh hoạt (Locking Knees)": "Thả lỏng đầu gối, làm động tác như chuẩn bị ngồi xuống ghế cao.",
            "Đầu gối khuỵu quá mức (Sitting too much)": "Đứng cao hơn chút, chỉ khuỵu nhẹ đầu gối để dồn trọng tâm lên ức bàn chân.",

            # Backswing & Top
            "Tay trái bị gập (Chicken Wing) - Mất bán kính swing": "Tập kẹp khăn dưới nách khi swing để giữ tay sát thân.",
            "Đầu di chuyển quá nhiều (Head Sway) - Gây mất ổn định": "Nhờ bạn giữ đầu hoặc nhìn vào gương, tập xoay vai mà đầu vẫn giữ yên.",
            "Hông trượt sang phải (Hip Slide) thay vì xoay": "Tập bài xoay hông trong xô nước (Barrel Turn) - tưởng tượng đứng trong thùng phuy và chỉ xoay, không trượt.",
            "Lỗi trục nghiêng ngược (Reverse Spine Angle) - Dễ gây đau lưng": "Đảm bảo vai trái hạ thấp hơn vai phải khi lên đỉnh, mắt vẫn nhìn vào phía sau bóng.",

            # Impact
            "Tay trái gập khi tiếp bóng (Scooping/Chicken Wing)": "Bài tập Impact Bag: Tập đánh vào túi đệm và dừng lại ở Impact để cảm nhận tay thẳng.",
            "Trọng tâm dồn về chân sau (Hanging Back) - Mất lực": "Bài tập bước chân (Step Drill): Bước chân trái lên khi downswing để ép trọng tâm chuyển sang trái.",
            "Thiếu chuyển trọng tâm sang chân trái": "Đặt bóng dưới lòng bàn chân phải, khi swing xuống phải nhấc gót phải lên khỏi bóng.",
            "Đầu lao về trước quá sớm (Lunging)": "Cố gắng cảm giác như 'đánh đầu' về phía sau khi gậy tiếp xúc bóng.",
            
            # Finish
            "Kết thúc không dồn hết trọng tâm sang trái": "Tập giữ tư thế kết thúc trong 3 giây, nhấc hoàn toàn chân phải lên chỉ tì bằng mũi giày."
        }

    def generate_report(self, diagnostic_json_path):
        with open(diagnostic_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        phases = data.get("phases", {})
        weighted_score = 0
        all_comments = []
        drills = set()
        
        # Tính toán điểm có trọng số
        for phase, analysis in phases.items():
            score = analysis.get("score", 0)
            weight = self.weights.get(phase, 0.1)
            weighted_score += score * weight
            
            # Thu thập nhận xét và drills
            for comment in analysis.get("comments", []):
                all_comments.append(f"[{phase}] {comment}")
                if comment in self.drill_library:
                    drills.add(self.drill_library[comment])

        # Phân loại trình độ dựa trên điểm (Thang điểm 10)
        score = round(weighted_score, 1)
        if score >= 9.0: level = "Professional / Low Handicap"
        elif score >= 7.5: level = "Mid Handicap"
        elif score >= 5.5: level = "High Handicap"
        else: level = "Beginner"

        final_report = {
            "video_id": data.get("video_id"),
            "final_score": score,
            "skill_level": level,
            "key_faults": all_comments,
            "recommended_drills": list(drills),
            "summary": self._generate_summary(score, all_comments)
        }
        
        return final_report

    def _generate_summary(self, score, faults):
        if not faults or faults[0].endswith("Đạt chuẩn"):
            return f"Cú swing của bạn đạt {score}/10. Kỹ thuật rất chuẩn, hãy tiếp tục duy trì!"
        
        summary = f"Điểm kỹ thuật của bạn đạt {score}/10. "
        if score < 7.0:
            summary += "Hệ thống phát hiện một số lỗi cần khắc phục để nâng cao hiệu suất. "
        else:
            summary += "Kỹ thuật của bạn khá tốt, chỉ cần tinh chỉnh một vài chi tiết nhỏ. "
            
        fault_text = faults[0].split('] ')[1] if faults else ""
        if fault_text != "Đạt chuẩn":
            summary += f"Lỗi ưu tiên cần sửa: {fault_text}."
        return summary

def generate_coaching_report_from_dict(analysis_dict):
    """
    Direct coaching report generation from analysis dict (optimized).
    
    Args:
        analysis_dict: Analysis result dictionary
    
    Returns:
        dict: Coaching report
    """
    coach = GolfCoachEngine()
    
    # Bypass file I/O - use dict directly
    phases = analysis_dict.get("phases", {})
    weighted_score = 0
    all_comments = []
    drills = set()
    
    # Tính toán điểm có trọng số
    for phase, analysis in phases.items():
        score = analysis.get("score", 0)
        weight = coach.weights.get(phase, 0.1)
        weighted_score += score * weight
        
        # Thu thập nhận xét và drills
        for comment in analysis.get("comments", []):
            all_comments.append(f"[{phase}] {comment}")
            if comment in coach.drill_library:
                drills.add(coach.drill_library[comment])

    # Phân loại trình độ
    score = round(weighted_score, 1)
    if score >= 9.0: level = "Professional / Low Handicap"
    elif score >= 7.5: level = "Mid Handicap"
    elif score >= 5.5: level = "High Handicap"
    else: level = "Beginner"

    final_report = {
        "video_id": analysis_dict.get("video_id"),
        "final_score": score,
        "skill_level": level,
        "key_faults": all_comments,
        "recommended_drills": list(drills),
        "summary": coach._generate_summary(score, all_comments)
    }
    
    return final_report

def generate_coaching_report(output_dir, return_dict=False):
    """
    Wrapper function để gọi từ main.py (tối ưu hóa).
    
    Args:
        output_dir: Thư mục output
        return_dict: True để return dict, False để ghi file
    
    Returns:
        dict: Báo cáo coaching
    """
    coach = GolfCoachEngine()
    
    # Đọc report.json (có thể từ dict hoặc file)
    input_path = os.path.join(output_dir, 'report.json')
    
    if os.path.exists(input_path):
        final_result = coach.generate_report(input_path)
        
        # Return dict nếu tối ưu hóa, hoặc ghi file (legacy)
        if return_dict:
            return final_result
        else:
            output_path = os.path.join(output_dir, 'FINAL_report.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)
            return final_result
    else:
        raise FileNotFoundError(f"Report file not found: {input_path}")

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Thư mục output của video')
    args = parser.parse_args()
    
    coach = GolfCoachEngine()
    
    input_path = os.path.join(args.output_dir, 'report.json')
    if os.path.exists(input_path):
        final_result = coach.generate_report(input_path)
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        
        # Lưu báo cáo cuối cùng vào thư mục output của video
        output_path = os.path.join(args.output_dir, 'FINAL_report.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        print(f"\nBáo cáo coaching cuối cùng đã lưu tại {output_path}")
    else:
        print(f"Lỗi: Không tìm thấy file: {input_path}")
        sys.exit(1)
