---
title: Golf Tech Analysis
emoji: 🏌️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Golf Tech Analysis AI

Hệ thống phân tích kỹ thuật Golf sử dụng AI (SwingNet & MediaPipe).

## Tính năng

- Phân tích 8 giai đoạn của cú Swing.
- Trích xuất dữ liệu khung xương (Skeleton) dưới dạng JSON.
- Phân tích tư thế và đưa ra nhận xét chuyên môn.
- Toàn bộ dữ liệu được xóa ngay sau khi xử lý (Stateless).

## Cách sử dụng

1. Tải video cú đánh của bạn lên qua giao diện Web.
2. Chờ hệ thống AI phân tích.
3. Tải về file `master_data.json` để sử dụng cho các công cụ phân tích hoặc render video.

## Deploy local

```bash
pip install -r requirements.txt
python api_server.py
```
