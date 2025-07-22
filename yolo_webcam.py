from ultralytics import YOLO
import cv2

# 1️⃣ Load model YOLOv8
model = YOLO('yolov8n.pt')

# 2️⃣ Mở webcam
cap = cv2.VideoCapture(0)

while True:
    # 3️⃣ Đọc 1 frame từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # 4️⃣ Dùng YOLO detect frame
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 5️⃣ Vẽ kết quả
    annotated_frame = results[0].plot() 

    # 6️⃣ Hiển thị
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # 7️⃣ Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8️⃣ Đóng mọi thứ
cap.release()
cv2.destroyAllWindows()
