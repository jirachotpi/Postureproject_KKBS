from ultralytics import YOLO
import cv2
import numpy as np

# ฟังก์ชันคำนวณมุม (หาความเอียงของคอ)
def calculate_neck_angle(ear, shoulder):
    # คำนวณระยะห่าง
    delta_x = ear[0] - shoulder[0]
    delta_y = ear[1] - shoulder[1]
    
    # คำนวณมุม (ใช้ arctan2) แล้วแปลงเป็นองศา
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.abs(angle_rad * 180.0 / np.pi)
    
    # แปลงให้เป็นมุมเทียบกับแนวตั้ง (90 องศา)
    # 90 คือแนวตั้งฉากเป๊ะๆ, ค่าที่ได้จะเป็นการเบี่ยงเบนจากแนวตั้ง
    neck_inclination = np.abs(90 - angle_deg)
    
    return neck_inclination

# โหลดโมเดล
model = YOLO('yolo11n-pose.pt') 
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot(boxes=False) # วาดโครงกระดูกให้

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for idx, kpts in enumerate(keypoints):
            if len(kpts) > 0:
                # YOLO Keypoints: 3=หูซ้าย, 5=ไหล่ซ้าย
                ear_x, ear_y = int(kpts[3][0]), int(kpts[3][1])
                shldr_x, shldr_y = int(kpts[5][0]), int(kpts[5][1])

                # คำนวณองศา "ความเอียงคอ"
                inclination = calculate_neck_angle((ear_x, ear_y), (shldr_x, shldr_y))

                # --- Logic เช็คหลังค่อม (Hunchback Logic) ---
                # เกณฑ์ตัดสิน (ปรับได้): 
                # < 15 องศา: คอตั้งตรง (หลังตรง)
                # > 25 องศา: คอเริ่มเอียงไปหน้าเยอะ (อาการหลังค่อม/ไหล่ห่อ)
                
                if inclination > 25:
                    status = "HUNCHBACK!"
                    color = (0, 0, 255) # แดง
                elif inclination > 15:
                    status = "Warning"
                    color = (0, 255, 255) # เหลือง
                else:
                    status = "Good Back"
                    color = (0, 255, 0) # เขียว

                # --- วาดภาพประกอบ ---
                # ลากเส้นจากไหล่ขึ้นไปหาหู
                cv2.line(annotated_frame, (shldr_x, shldr_y), (ear_x, ear_y), color, 3)
                # ลากเส้นแนวตั้งเปรียบเทียบ (สีขาวบางๆ)
                cv2.line(annotated_frame, (shldr_x, shldr_y), (shldr_x, shldr_y - 100), (200, 200, 200), 1)

                # แสดงข้อความ
                cv2.putText(annotated_frame, f"{int(inclination)} deg", 
                           (shldr_x + 10, shldr_y - 40), font, 0.6, color, 2)
                
                cv2.putText(annotated_frame, status, 
                           (shldr_x - 40, shldr_y - 70), font, 0.8, color, 2)

    cv2.imshow("YOLOv11 - Hunchback Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()