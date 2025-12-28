---
title: Golf Tech Analysis
emoji: 🏌️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Golf Tech Analysis

Phân tích kỹ thuật golf swing bằng AI - Tự động nhận diện 8 giai đoạn, chấm điểm và đưa ra coaching.

## Tính năng

- Nhận diện 8 giai đoạn swing (SwingNet AI)
- Phân tích tư thế (MediaPipe Pose)
- Chấm điểm tự động (0-10 điểm)
- Đề xuất bài tập khắc phục

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### CLI - Lưu kết quả ra file

```bash
python main.py video.mp4
# Kết quả: results/[video_id]/master_data.json
# Video overlay: results/[video_id]/analyzed_video.mp4
```

### API Server

```bash
python api_server.py
# Server: http://localhost:7860
```

**Endpoints:**

- `GET /` - Giao diện test upload
- `POST /` - Tạo video có overlay (trả về file .mp4)
- `POST /api/analyze` - API endpoint (chỉ trả JSON)

### Tạo video có overlay từ JSON

```bash
python reengineer.py --json results/[id]/master_data.json --video video.mp4 --output output.mp4
```

## Kết quả JSON

```json
{
  "coaching": {
    "final_score": 9.2,
    "skill_level": "Professional",
    "key_faults": [...],
    "recommended_drills": [...]
  }
}
```

## Deploy

- **GitHub**: https://github.com/htrnguyen/golf-tech-analysis
- **Hugging Face**: https://huggingface.co/spaces/htrnguyen/golf-tech-analysis

---

Made by htrnguyen
