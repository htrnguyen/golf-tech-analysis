# Sử dụng Python 3.10 slim để dung lượng nhẹ hơn
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo người dùng không có quyền root (Yêu cầu của Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy các dependencies trước để tối ưu hóa Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy toàn bộ code vào container
COPY --chown=user . .

# Hugging Face Spaces lắng nghe trên cổng 7860 mặc định
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
