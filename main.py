import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import shutil

sys.path.insert(0, os.path.abspath('.'))
from extract import run_ai_extraction
from analyze import GolfDiagnosticEngine
from coach import generate_coaching_report_from_dict

def analyze_video_fast(video_path, production=True, output_file=None, output_base="output"):
    """
    Phiên bản tối ưu: Gọi trực tiếp các function thay vì subprocess.
    
    Args:
        video_path: Đường dẫn video
        production: Nếu True, tắt logs để tăng tốc
        output_file: Nếu có, lưu kết quả vào file này (CLI mode)
        output_base: Thư mục gốc để lưu output ("results" cho CLI, "output" cho API)
    
    Returns:
        dict: Master data JSON
    """
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    
    output_dir = os.path.join(output_base, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if not production:
        print(f"Analyzing: {video_name}")
    
    try:
        extraction_result = run_ai_extraction(
            video_path, 
            slow_factor=1.0, 
            output_dir=output_dir,
            skip_slow_video=True,
            skip_phase_images=False,
            return_dict=True,
            production=production
        )
        
        if not extraction_result:
            raise Exception("Extraction failed")
        

        engine = GolfDiagnosticEngine()
        analysis_result = engine.process_video_results(
            output_dir,
            return_dict=True  # NEW: Return dict thay vì ghi file
        )
        
        if not analysis_result:
            raise Exception("Analysis failed")
        
        # Step 3: Generate coaching (pass analysis result as dict)
        coaching_result = generate_coaching_report_from_dict(analysis_result)
        
        if not coaching_result:
            raise Exception("Coaching failed")
        
        # Cleanup phases directory ngay lập tức
        phases_dir = os.path.join(output_dir, 'phases')
        if os.path.exists(phases_dir):
            shutil.rmtree(phases_dir, ignore_errors=True)
        
        # Tổng hợp master data
        master_data = {
            "status": "success",
            "job_id": video_id,
            "metadata": extraction_result.get("metadata", {}),
            "analysis": analysis_result,
            "coaching": coaching_result
        }
        
        # Lưu file nếu CLI mode
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(master_data, f, indent=4, ensure_ascii=False)
            if not production:
                print(f"\nKết quả đã lưu tại: {output_file}")
            
            # CLI mode: Tự động tạo video có overlay
            if not production:
                try:
                    from reengineer import reengineer_video
                    output_video = os.path.join(os.path.dirname(output_file), 'analyzed_video.mp4')
                    print(f"\nĐang tạo video có overlay...")
                    reengineer_video(output_file, video_path, output_video, production=production)
                    print(f"Video đã lưu tại: {output_video}")
                except Exception as e:
                    print(f"WARNING: Không thể tạo video overlay: {e}")
        
        # Cleanup ONLY nếu không phải CLI mode (API mode)
        if not output_file:
            # Cleanup phases directory
            phases_dir = os.path.join(output_dir, 'phases')
            if os.path.exists(phases_dir):
                shutil.rmtree(phases_dir, ignore_errors=True)
            # Cleanup output directory sau khi có data
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            # CLI mode: Chỉ cleanup phases, giữ lại master_data.json
            phases_dir = os.path.join(output_dir, 'phases')
            if os.path.exists(phases_dir):
                shutil.rmtree(phases_dir, ignore_errors=True)
        
        return master_data
        
    except Exception as e:
        if not production:
            import traceback
            traceback.print_exc()
        
        # Cleanup on error
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        
        raise e

# CLI function
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hệ thống Phân tích Golf')
    parser.add_argument('video_path', help='Đường dẫn đến file video .mp4')
    args = parser.parse_args()
    
    video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    output_file = os.path.join('results', video_id, 'master_data.json')
    
    result = analyze_video_fast(args.video_path, production=False, output_file=output_file, output_base="results")
    # Kết quả đã được in ra và lưu file

if __name__ == "__main__":
    main()
