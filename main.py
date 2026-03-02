import threading
import time
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# นำเข้า Class ที่เรา Refactor ไว้ในไฟล์ posture_engine.py
from posture_engine import PostureMonitorApp

app = FastAPI(title="ErgoSide Posture API")

# ปรับปรุง CORS เพื่อให้ React (ปกติ port 3000 หรือ 5173) ติดต่อได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้าง Instance ของ Engine
monitor = PostureMonitorApp()
current_processed_frame = None
frame_lock = threading.Lock()

def cv_background_thread():
    """Thread สำหรับจัดการกล้องและส่งภาพให้ Engine ประมวลผล"""
    global current_processed_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # ส่งเฟรมไปให้ Engine ประมวลผล (Refactored method)
        processed = monitor.process_frame(frame)

        with frame_lock:
            current_processed_frame = processed.copy()
            
    cap.release()

# --- API ENDPOINTS ---

@app.get("/status")
async def get_status():
    """Endpoint สำหรับ React มาดึงข้อมูล Real-time ไปโชว์"""
    return {
        "state": monitor.current_state,
        "fhp_ratio": round(monitor.current_fhp_ratio, 3),
        "slump_angle": round(monitor.current_slump_angle, 1),
        "is_calibrated": monitor.baseline_torso is not None,
        "is_standing": monitor.is_user_standing
    }

@app.post("/calibrate")
async def trigger_calibration():
    """ปุ่มกด Calibrate จากหน้า Dashboard"""
    monitor.start_calibration()
    return {"status": "success", "message": "Calibration started for 5 seconds"}

@app.get("/statistics")
async def get_stats():
    """ดึงข้อมูล JSON ที่เก็บสะสมไว้รายวัน"""
    return monitor.stats.data

@app.get("/video_feed")
async def video_feed():
    """Streaming ภาพจากกล้องพร้อม Landmark ไปที่หน้าเว็บ"""
    def generate():
        while True:
            with frame_lock:
                if current_processed_frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', current_processed_frame)
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04) # จำกัดไว้ที่ประมาณ 25 FPS เพื่อประหยัด CPU

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # เริ่มต้น Background Thread สำหรับประมวลผลภาพ
    t = threading.Thread(target=cv_background_thread, daemon=True)
    t.start()
    
    # รัน API Server (Port 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)