import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import subprocess
import argparse

def run_step(command, step_desc):
    """
    Thực thi một lệnh shell và in kết quả gọn gàng.
    """
    print(f"{step_desc}...", end=" ", flush=True)
    
    # Force UTF-8 encoding for subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(command, capture_output=True, text=True, errors='replace', encoding='utf-8', env=env)
    
    if result.returncode != 0:
        print("FAILED")
        print(f"Lỗi chi tiết:\n{result.stderr}")
        return False
        
    print("DONE")
    return True

def main():
    # Cấu hình stdout để hỗ trợ tiếng Việt trên Windows terminal
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description='Hệ thống Phân tích Golf thông minh (End-to-End)')
    parser.add_argument('video_path', help='Đường dẫn đến file video .mp4')
    parser.add_argument('--slow', type=float, default=1.0, help='Tỷ lệ làm chậm (0.5 hoặc 0.2 để tăng độ chính xác)')
    args = parser.parse_args()

    video_path = args.video_path
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    
    # Tạo thư mục output riêng cho video này
    output_dir = os.path.join('output', video_id)
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================")
    print(f"BẮT ĐẦU PHÂN TÍCH VIDEO: {video_name}")
    print(f"Thư mục kết quả: {output_dir}")
    print("====================================================")

    # Bước 1: Trích xuất giai đoạn bằng AI
    if not run_step([sys.executable, 'extract.py', video_path, '--slow', str(args.slow), '--output_dir', output_dir], 
                    "[1/4] Nhận diện giai đoạn"):
        sys.exit(1)

    # Bước 2: Chẩn đoán tư thế (Landmark Analysis)
    if not run_step([sys.executable, 'analyze.py', output_dir], 
                    "[2/4] Phân tích tư thế"):
        sys.exit(1)
        
    # Tự động dọn dẹp thư mục phases để tiết kiệm dung lượng
    import shutil
    phases_dir = os.path.join(output_dir, 'phases')
    if os.path.exists(phases_dir):
        try:
            shutil.rmtree(phases_dir)
        except Exception as e:
            pass # Silent cleanup

    # Bước 3: Đưa ra nhận xét và chấm điểm (Coaching Engine)
    if not run_step([sys.executable, 'coach.py', output_dir], 
                    "[3/4] Chấm điểm & Phân tích"):
        sys.exit(1)

    # Bước 4: Tạo video báo cáo trực quan (Timeline) - ĐÃ LOẠI BỎ ĐỂ CHUYỂN SANG OVERLAY PHÍA CLIENT
    # slow_video_path = os.path.join(output_dir, "slow_motion.mp4")
    # target_video_for_report = video_path
    # if os.path.exists(slow_video_path):
    #     target_video_for_report = slow_video_path
    # if not run_step([sys.executable, 'report.py', target_video_for_report, output_dir], 
    #                 "[4/4] Tạo báo cáo video"):
    #     sys.exit(1)

    # 4. TỔNG HỢP VÀ DỌN DẸP -> CHỈ GIỮ LẠI MASTER JSON
    final_report_path = os.path.join(output_dir, "FINAL_report.json")
    if os.path.exists(final_report_path):
        try:
            with open(final_report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            master_data = {
                "status": "success",
                "job_id": video_id,
                "metadata": {},
                "analysis": {},
                "coaching": report_data
            }
            
            # Đọc Metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    master_data["metadata"] = json.load(f)
            
            # Đọc Analysis chi tiết
            report_detail_path = os.path.join(output_dir, "report.json")
            if os.path.exists(report_detail_path):
                with open(report_detail_path, 'r', encoding='utf-8') as f:
                    master_data["analysis"] = json.load(f)
            
            # Lưu Master JSON
            master_json_path = os.path.join(output_dir, "master_data.json")
            with open(master_json_path, 'w', encoding='utf-8') as f:
                json.dump(master_data, f, indent=4, ensure_ascii=False)
            
            # --- DỌN DẸP FILE TRUNG GIAN (CHỈ GIỮ MASTER JSON) ---
            annotated_dir = os.path.join(output_dir, "annotated")
            files_to_delete = [metadata_path, report_detail_path, final_report_path]
            
            # Xóa các file JSON trung gian
            for f_path in files_to_delete:
                if os.path.exists(f_path):
                    os.remove(f_path)
            
            # Xóa thư mục annotated nếu tồn tại
            if os.path.exists(annotated_dir):
                shutil.rmtree(annotated_dir, ignore_errors=True)
            
            print(f"\n[MASTER_JSON_CREATED]: {master_json_path}")
            
        except Exception as e:
            print(f"Lưu Master JSON hoặc Dọn dẹp thất bại: {e}")
    else:
        print(f"\n[ERROR]: Không tìm thấy FINAL_report.json để tạo Master JSON.")

if __name__ == "__main__":
    main()
