# 🏌️ Golf Tech Analysis - Hệ Thống Phân Tích Kỹ Thuật Golf AI

Hệ thống phân tích kỹ thuật golf sử dụng AI (SwingNet & MediaPipe) để nhận diện 8 giai đoạn swing, phân tích tư thế và đưa ra coaching chuyên nghiệp.

## ✨ Tính Năng

- 🎯 **Phân tích 8 giai đoạn swing**: Address, Toe-up, Mid-Backswing, Top, Mid-Downswing, Impact, Mid-Follow-Through, Finish
- 🧠 **AI Event Detection**: Sử dụng SwingNet để nhận diện chính xác từng giai đoạn
- 💪 **Pose Analysis**: MediaPipe phân tích độ chính xác tư thế từng giai đoạn
- 📊 **Scoring System**: Chấm điểm có trọng số, Impact chiếm 35%
- 🎓 **Coaching Engine**: Đưa ra nhận xét và gợi ý bài tập khắc phục
- ⚡ **Optimized Pipeline**: Xử lý nhanh, không subprocess overhead
- 🔒 **Privacy First**: API mode xóa sạch dữ liệu ngay sau khi xử lý

## 🚀 Cài Đặt

### Requirements

- Python 3.10+
- CUDA (optional, để tăng tốc)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Sử Dụng

### 1. CLI Mode (Lưu kết quả vào file)

```bash
python main.py video.mp4
```

Kết quả sẽ được lưu vào: `results/[video_id]/master_data.json`

### 2. API Server

Khởi động server:

```bash
python api_server.py
```

Server sẽ chạy tại: `http://localhost:7860`

#### Endpoints:

- **GET `/`** - Giao diện upload để test (drag & drop)
- **POST `/api/analyze`** - Pure API endpoint cho app/web

#### API Usage Example:

```python
import requests

url = "http://localhost:7860/api/analyze"
with open("video.mp4", "rb") as f:
    response = requests.post(url, files={"file": f})
    result = response.json()

print(f"Score: {result['coaching']['final_score']}/10")
print(f"Level: {result['coaching']['skill_level']}")
```

### 3. Offline Video Re-engineering

Tạo video có skeleton overlay và coaching comments:

```bash
python reengineer.py --json master_data.json --video video.mp4 --output analyzed.mp4
```

## 📁 Cấu Trúc Dữ Liệu

### Master Data JSON

```json
{
  "status": "success",
  "job_id": "video_id",
  "metadata": {
    "event_frames": { "1_Address": 10, "6_Impact": 62, ... },
    "fps": 30,
    "slow_factor": 1.0
  },
  "analysis": {
    "phases": {
      "1_Address": { "score": 8.5, "comments": [...] },
      ...
    },
    "overall_score": 9.2,
    "view_angle": "Face-on"
  },
  "coaching": {
    "final_score": 9.2,
    "skill_level": "Professional / Low Handicap",
    "key_faults": [...],
    "recommended_drills": [...]
  }
}
```

## 🎨 Offline Rendering (reengineer.py)

Tool này cho phép bạn tái tạo video với overlay từ `master_data.json`:

- ✅ Skeleton overlay (31 điểm MediaPipe)
- ✅ Phase labels và freeze-frames
- ✅ Coaching comments
- ✅ Score hiển thị
- ✅ Overwrite mode hoặc tạo file mới

```bash
# Overwrite video gốc
python reengineer.py --json results/video_id/master_data.json --video video.mp4

# Tạo file mới
python reengineer.py --json results/video_id/master_data.json --video video.mp4 --output analyzed.mp4
```

## 🏗️ Kiến Trúc

### Pipeline

```
Video Input
   ↓
extract.py (SwingNet AI) → Event frames
   ↓
analyze.py (MediaPipe) → Pose analysis + Scoring
   ↓
coach.py (Coaching Engine) → Final recommendations
   ↓
master_data.json
```

### Performance Optimizations

- ❌ No subprocess spawning (direct imports)
- ❌ No slow-motion video generation in API mode
- ❌ No intermediate file I/O
- ✅ Dict-based pipeline communication
- ✅ Immediate cleanup after response

**Result**: 60-70% faster (từ ~10-15s → ~3-5s)

## 📦 Deployment

### Hugging Face Spaces

Project đã được tối ưu để chạy trên HF Spaces với Docker:

- `Dockerfile` đã cấu hình OpenCV + MediaPipe
- Port 7860 (HF requirement)
- Git LFS cho model files (\*.pth.tar)

### Local Development

```bash
# CLI
python main.py video.mp4

# API Server
python api_server.py
```

## 📊 Scoring Methodology

Xem chi tiết tại: [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)

- **Impact**: 35% (quan trọng nhất)
- **Top**: 20%
- **Address**: 10%
- **Finish**: 10%
- **Mid-Downswing**: 10%
- **Các phase khác**: 5% mỗi phase

## 🤖 Models

- **SwingNet**: Event detection (8 golf swing events)
- **MediaPipe Pose**: 33-point skeleton tracking
- Models được lưu trong `models/`:
  - `swingnet_1800.pth.tar` (63MB)
  - `mobilenet_v2.pth.tar` (14MB)

## 📝 License

MIT License - Sử dụng tự do cho mục đích cá nhân và thương mại.

## 🙏 Credits

- SwingNet: Golf swing event detection
- MediaPipe: Real-time pose estimation
- GolfDB: Dataset inspiration

---

Made with ❤️ by htrnguyen
