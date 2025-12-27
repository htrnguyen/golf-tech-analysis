import os
import shutil
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import subprocess
import sys

app = FastAPI(title="Golf Tech Analysis API - Stateless")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
TEMPLATES_DIR = "templates"

# Đảm bảo các thư mục tồn tại (chúng sẽ chỉ chứa file tạm thời cực ngắn)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.post("/")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Tạo job tạm thời
    job_id = str(uuid.uuid4())
    video_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")

    try:
        # 2. Lưu video tạm thời
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Pipeline Error: {process.stderr}")

        # 4. Đọc kết quả Master JSON
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        job_output_dir = os.path.join(OUTPUT_DIR, video_id)
        master_json_path = os.path.join(job_output_dir, "master_data.json")
        
        if not os.path.exists(master_json_path):
            raise HTTPException(status_code=500, detail="Không tìm thấy master_data.json.")

        with open(master_json_path, 'r', encoding='utf-8') as f:
            master_report = json.load(f)

        # 5. DỌN DẸP TUYỆT ĐỐI NGAY LẬP TỨC (Không lưu lại bất cứ gì)
        shutil.rmtree(job_output_dir, ignore_errors=True)
        if os.path.exists(video_path): os.remove(video_path)
        
        # Xóa các file rác khác trong uploads nếu có
        # (Chỉ xóa file của chính job này để đảm bảo concurrency)
        
        # Trả về JSON Data trực tiếp (Browser sẽ nhận được text/json)
        return master_report

    except Exception as e:
        # Cleanup nếu có lỗi
        if os.path.exists(video_path): os.remove(video_path)
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        shutil.rmtree(os.path.join(OUTPUT_DIR, video_id), ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces yêu cầu cổng 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
