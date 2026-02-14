from ultralytics import YOLO
import cv2
import time

# ใช้รุ่น Nano (n) เพื่อความเร็วสูงสุดบน CPU
model = YOLO('yolo11n-pose.pt') 

cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    
    # YOLO คำนวณ
    results = model(frame, verbose=False) # verbose=False ปิด log รกๆ ใน terminal
    annotated_frame = results[0].plot()

    # คำนวณ FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # แสดงค่า FPS ตัวใหญ่ๆ สีฟ้า
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cv2.imshow('YOLOv11 Speed Test', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()