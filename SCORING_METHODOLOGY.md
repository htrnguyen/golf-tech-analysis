# Thuyết minh: Phương pháp Đánh giá Golf AI (Thang điểm 10)

Hệ thống của chúng tôi đánh giá cú swing dựa trên sự kết hợp giữa **Dữ liệu Hình ảnh (AI)** và **Hình học Tư thế (Pose Geometry)**.

## 1. Cơ chế Chấm điểm

- **Điểm cơ sở**: Mỗi giai đoạn bắt đầu với **10.0 điểm**.
- **Điểm trừ (Penalties)**: Tùy theo mức độ nghiêm trọng của lỗi, hệ thống sẽ trừ từ 0.5 đến 2.5 điểm mỗi lỗi.
- **Trọng số (Weights)**: Điểm tổng kết không phải là trung bình cộng đơn thuần mà được tính theo tầm quan trọng của từng pha.

| Giai đoạn    | Trọng số | Tầm quan trọng                           |
| :----------- | :------- | :--------------------------------------- |
| Address      | 10%      | Nền tảng ban đầu                         |
| Top          | 20%      | Đỉnh cao của năng lượng                  |
| Impact       | 35%      | **Quan trọng nhất** (Điểm tiếp xúc bóng) |
| Finish       | 10%      | Thể hiện sự thăng bằng                   |
| Các pha khác | 25%      | Độ mượt mà của quá trình chuyển động     |

## 2. Tiêu chí Chẩn đoán Chi tiết

### Pha Address (Tư thế chuẩn bị)

- **Độ rộng chân (Stance Width)**: Được đo so với độ rộng vai.
  - _Chuẩn_: Tỉ lệ từ 0.8 đến 1.4.
  - _Lỗi_: Quá rộng (mất linh hoạt) hoặc quá hẹp (mất thăng bằng).

### Pha Top (Đỉnh Backswing)

- **Tay dẫn (Lead Arm - Tay trái)**: Phải thẳng để tạo cung swing lớn nhất.
  - _Chuẩn_: Góc tay >= 150 độ.
  - _Lỗi_: "Chicken Wing" (Tay bị gập) làm giảm lực đánh.

### Pha Impact (Tiếp bóng)

- **Tay dẫn lúc va chạm**: Đây là khoảnh khắc truyền lực.
  - _Chuẩn_: Góc tay >= 160 độ (Gần như thẳng tuyệt đối).
  - _Lỗi_: Tay bị nhấc hoặc gập sớm làm mất lực và dễ đánh trượt.

### Pha Finish (Kết thúc)

- **Thăng bằng (Balance Control)**: Kiểm tra vị trí hông so với chân trụ.
  - _Chuẩn_: Trọng tâm dồn hoàn toàn về chân trái, kết thúc vững chãi.
  - _Lỗi_: Ngả người ra sau hoặc đổ về phía trước.

## 3. Phân loại Trình độ

- **9.0 - 10.0**: Pro / Low Handicap (Kỹ thuật tiệm cận chuyên nghiệp).
- **7.5 - 8.9**: Mid Handicap (Kỹ thuật tốt, cần tinh chỉnh chi tiết).
- **5.5 - 7.4**: High Handicap (Cần tập trung vào các bài tập cơ bản).
- **Dưới 5.5**: Beginner (Cần sự hướng dẫn trực tiếp từ huấn luyện viên).
