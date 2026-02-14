import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ตั้งค่าโมเดล
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Program Started... กด 'q' เพื่อออก")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # 1. เตรียมภาพ
    frame = cv2.flip(frame, 1) # กลับด้านกระจก
    h, w, c = frame.shape      # เก็บขนาดภาพ
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 2. ถ้าเจอคน
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- ดึงพิกัดจุดสำคัญ ---
        # จุดที่ 0: จมูก
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        cx_nose, cy_nose = int(nose.x * w), int(nose.y * h)

        # จุดที่ 11: ไหล่ซ้าย
        l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        cx_l_shldr, cy_l_shldr = int(l_shldr.x * w), int(l_shldr.y * h)

        # จุดที่ 12: ไหล่ขวา
        r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        cx_r_shldr, cy_r_shldr = int(r_shldr.x * w), int(r_shldr.y * h)

        # --- วาดลงบนภาพ ---
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # วาดจุด
        cv2.circle(frame, (cx_nose, cy_nose), 10, (0, 0, 255), -1)
        cv2.circle(frame, (cx_l_shldr, cy_l_shldr), 10, (0, 255, 0), -1)
        cv2.circle(frame, (cx_r_shldr, cy_r_shldr), 10, (0, 255, 0), -1)

        # เขียนพิกัด
        cv2.putText(frame, f"Nose: {cx_nose},{cy_nose}", (cx_nose+15, cy_nose), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        # คำนวณ Posture (ตัวอย่างง่ายๆ)
        shoulder_avg_y = (cy_l_shldr + cy_r_shldr) / 2
        
        # เงื่อนไข: ถ้าจมูกต่ำลงมาใกล้แนวไหล่ (คอยื่น/ก้มหน้า)
        if cy_nose > shoulder_avg_y - 80: 
            # สีแดง: ท่าไม่ดี
            cv2.putText(frame, "BAD POSTURE!", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5) # เลข 5 ข้างหลังคือความหนาตัวอักษร
        else:
            # สีเขียว: ท่าดี
            cv2.putText(frame, "Good Posture", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    cv2.imshow('PostureTag - Key Points', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()