import threading
import time
import cv2
import uvicorn
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from posture_engine import PostureMonitorApp

app = FastAPI(title="ErgoSide Posture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = PostureMonitorApp()
current_processed_frame = None
frame_lock = threading.Lock()


def to_python(obj):
    """แปลง numpy type / object แปลก ๆ ให้เป็น JSON-safe"""
    if obj is None:
        return None

    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def cv_background_thread():
    global current_processed_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        processed = monitor.process_frame(frame)

        with frame_lock:
            current_processed_frame = processed.copy()

    cap.release()


@app.get("/status")
async def get_status():
    try:
        data = {
            "state": str(to_python(monitor.current_state)),
            "fhp_ratio": round(float(to_python(monitor.current_fhp_ratio) or 0.0), 3),
            "slump_angle": round(float(to_python(monitor.current_slump_angle) or 0.0), 1),
            "is_calibrated": bool(monitor.baseline_torso is not None),
            "is_standing": bool(to_python(monitor.is_user_standing)),
        }

        return JSONResponse(
            content=to_python(data),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/calibrate")
async def trigger_calibration():
    monitor.start_calibration()
    return {"status": "success", "message": "Calibration started for 5 seconds"}


@app.get("/statistics")
async def get_stats():
    return JSONResponse(content=to_python(monitor.stats.data))


@app.get("/history")
async def get_history():
    """ดึงข้อมูลประวัติทุกวันจาก MongoDB"""
    try:
        # ดึงเอกสารทั้งหมดจาก collection
        cursor = monitor.stats.collection.find({})
        raw_list = list(cursor)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    result = {}
    for day in raw_list:
        date_id = day.get("_id")
        total = day.get("total_sitting_seconds", 0)
        states = day.get("states", {})

        # คำนวณ score และ breakdown % อัตโนมัติเหมือนเดิม
        good_sec = states.get("GOOD", 0)
        ergonomic_score = round((good_sec / total * 100) if total > 0 else 0, 1)

        breakdown = {}
        for state, sec in states.items():
            breakdown[state] = round((sec / total * 100) if total > 0 else 0, 1)

        result[date_id] = {
            "total_sitting_seconds": total,
            "states": states,
            "ergonomic_score": ergonomic_score,
            "breakdown": breakdown,
            "last_updated": day.get("last_updated")
        }

    return JSONResponse(content=to_python(result))


@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_processed_frame is None:
                    continue
                _, buffer = cv2.imencode(".jpg", current_processed_frame)
                frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    t = threading.Thread(target=cv_background_thread, daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)