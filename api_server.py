import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Golf Tech Analysis API")

UPLOAD_DIR = "uploads"
TEMPLATES_DIR = "templates"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Import optimized pipeline
from main import analyze_video_fast
from reengineer import reengineer_video

@app.get("/")
async def root(request: Request):
    """
    Giao diện upload để test (cho developer).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def analyze_with_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    UI endpoint: Phân tích + Tạo video có overlay.
    Trả về video để download.
    """
    job_id = str(uuid.uuid4())
    video_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")
    output_dir = os.path.join("output", job_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Lưu video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Phân tích (lưu vào output/)
        master_json = os.path.join(output_dir, "master_data.json")
        result = analyze_video_fast(video_path, production=True, output_file=master_json, output_base="output")
        
        # Tạo video có overlay
        output_video = os.path.join(output_dir, "analyzed_video.mp4")
        reengineer_video(master_json, video_path, output_video, production=True)
        
        # Cleanup video gốc
        if os.path.exists(video_path): 
            os.remove(video_path)
        
        # Schedule cleanup after response
        def cleanup():
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass
        
        if background_tasks:
            background_tasks.add_task(cleanup)
        
        # Trả về video file
        return FileResponse(
            output_video, 
            media_type="video/mp4",
            filename=f"golf_analysis_{job_id}.mp4"
        )

    except Exception as e:
        if os.path.exists(video_path): 
            os.remove(video_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    """
    Pure API endpoint: Chỉ trả JSON, không tạo video.
    Dùng cho app/web production.
    """
    job_id = str(uuid.uuid4())
    video_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")

    try:
        # Lưu video tạm thời
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Gọi optimized pipeline (KHÔNG tạo video)
        result = analyze_video_fast(video_path, production=True)
        
        # Cleanup
        if os.path.exists(video_path): 
            os.remove(video_path)
        
        return result

    except Exception as e:
        if os.path.exists(video_path): 
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
