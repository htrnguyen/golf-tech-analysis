import os
import shutil
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import sys

app = FastAPI(title="Golf Tech Analysis API - Pure Backend")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

# Đảm bảo các thư mục tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/")
async def analyze_video(file: UploadFile = File(...)):
    """
    Endpoint chính: Nhận video, phân tích và trả về Master JSON.
    """
    # 1. Tạo job tạm thời
    job_id = str(uuid.uuid4())
    video_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")

    try:
        # 1.5 Kiểm tra môi trường (Diagnostics for HF Logs)
        print(f"DEBUG: WORKDIR: {os.getcwd()}")
        if os.path.exists('models'):
            print(f"DEBUG: Models found: {os.listdir('models')}")

        # 2. Lưu video tạm thời
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"DEBUG: Video saved ({os.path.getsize(video_path)} bytes)")

        # 3. Chạy Pipeline phân tích
        base_path = os.path.dirname(os.path.abspath(__file__))
        main_py_path = os.path.join(base_path, "main.py")
        video_abs_path = os.path.abspath(video_path)
        
        process = subprocess.run(
            [sys.executable, main_py_path, video_abs_path],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        # Log output (quan trọng để debug trên Hugging Face)
        if process.stdout: print(f"STDOUT: {process.stdout}")
        if process.stderr: print(f"STDERR: {process.stderr}")

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Pipeline Error: {process.stderr}")

        # 4. Đọc kết quả Master JSON
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        job_output_dir = os.path.join(OUTPUT_DIR, video_id)
        master_json_path = os.path.join(job_output_dir, "master_data.json")
        
        if not os.path.exists(master_json_path):
            raise HTTPException(status_code=500, detail="Analysis failed: master_data.json not found.")

        with open(master_json_path, 'r', encoding='utf-8') as f:
            master_report = json.load(f)

        # 5. DỌN DẸP TUYỆT ĐỐI NGAY LẬP TỨC
        shutil.rmtree(job_output_dir, ignore_errors=True)
        if os.path.exists(video_path): os.remove(video_path)
        
        # Trả về kết quả JSON trực tiếp
        return master_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Cleanup
        if os.path.exists(video_path): os.remove(video_path)
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        shutil.rmtree(os.path.join(OUTPUT_DIR, video_id), ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces yêu cầu cổng 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
