import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import cv2

from posture_engine import PostureMonitorApp

app = FastAPI(title="ErgoSide Posture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Object สำหรับแชร์ข้อมูลระหว่าง Thread
monitor = PostureMonitorApp()
current_frame = None
lock = threading.Lock()

def run_cv_engine():
    """Background Thread สำหรับรัน Computer Vision"""
    global current_frame
    cap = cv2.VideoCapture(0)
    # ตั้งค่าอื่นๆ ตามที่ออกแบบไว้
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # ประมวลผล Posture
        processed_frame = monitor.process_frame(frame) # ปรับ Logic เดิมให้รับ/ส่ง frame
        
        with lock:
            current_frame = processed_frame
            
    cap.release()

# ------------------------------
# API Endpoints
# ------------------------------

@app.get("/status")
async def get_status():
    """ดึงสถานะ Posture ปัจจุบัน และ Metrics ต่างๆ"""
    return {
        "state": monitor.state_machine.state,
        "fhp_ratio": monitor.current_fhp_ratio,
        "slump_angle": monitor.current_slump_angle,
        "is_calibrated": monitor.baseline_torso is not None,
        "is_standing": monitor.is_user_standing
    }

@app.post("/calibrate")
async def trigger_calibration():
    """สั่งเริ่มการ Calibrate จากหน้าเว็บ"""
    monitor.start_calibration()
    return {"message": "Calibration started"}

@app.get("/statistics")
async def get_daily_stats():
    """ดึงข้อมูลสถิติจากไฟล์ JSON"""
    return monitor.stats.data

@app.get("/video_feed")
async def video_feed():
    """Streaming Video ไปแสดงบน Web Dashboard (MJPEG)"""
    def generate():
        while True:
            with lock:
                if current_frame is None:
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", current_frame)
                if not flag:
                    continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03) # ประมาณ 30 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ------------------------------
# Start Server
# ------------------------------
if __name__ == "__main__":
    cv_thread = threading.Thread(target=run_cv_engine, daemon=True)
    cv_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)