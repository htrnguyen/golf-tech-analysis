import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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

@app.get("/")
async def root(request: Request):
    """
    Giao diện upload để test (cho developer).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    """
    Pure API endpoint: Nhận video, trả về JSON.
    Dùng cho app/web production.
    """
    job_id = str(uuid.uuid4())
    video_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")

    try:
        # Lưu video tạm thời
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Gọi optimized pipeline
        result = analyze_video_fast(video_path, production=True)
        
        # Cleanup
        if os.path.exists(video_path): 
            os.remove(video_path)
        
        return result

    except Exception as e:
        if os.path.exists(video_path): 
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
