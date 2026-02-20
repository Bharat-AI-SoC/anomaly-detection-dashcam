import onnxruntime as ort
import numpy as np
import cv2
import time
import datetime
import os
import threading
from picamera2 import Picamera2

# --- CONFIG ---
MODEL_PATH = "best_int8.onnx"
INPUT_SIZE = (416, 416)
CONF_THRESHOLD = 0.5
BUFFER_SIZE = 30 
DISPLAY_REDUCE = 2 # Show every 2nd frame to save CPU

# Optimized Session
opts = ort.SessionOptions()
opts.add_session_config_entry("session.use_xnnpack", "1")
opts.intra_op_num_threads = 4 
session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

os.makedirs("clips", exist_ok=True)

# PRE-ALLOCATED BUFFER
video_buffer = np.zeros((BUFFER_SIZE, 480, 640, 3), dtype=np.uint8)
buffer_index = 0
save_event = threading.Event()
latest_detections = None

def background_saver():
    global latest_detections
    while True:
        save_event.wait()
        # Capture buffer in correct chronological order
        frames_to_save = np.concatenate(
            (video_buffer[buffer_index:], video_buffer[:buffer_index]),
            axis=0
        )
        dets = latest_detections
        
        ts = datetime.datetime.now().strftime("%H%M%S")
        fname = f"clips/pothole_{ts}.mp4"
        out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))
        
        for f in frames_to_save:
            # We still draw boxes for the saved clip!
            if dets is not None:
                for det in dets[0][0]:
                    conf = det[4]
                    if conf > CONF_THRESHOLD:

                        # YOLO format: x_center, y_center, w, h
                        xc, yc, w, h = det[:4]

                        # Convert to corner format in 416 space
                        x1 = (xc - w/2)
                        y1 = (yc - h/2)
                        x2 = (xc + w/2)
                        y2 = (yc + h/2)

                        # Scale to 640x480
                        x1 = int(x1 * 640 / 416)
                        x2 = int(x2 * 640 / 416)
                        y1 = int(y1 * 480 / 416)
                        y2 = int(y2 * 480 / 416)

                        cv2.rectangle(f, (x1, y1), (x2, y2), (0,0,255), 2)
            out.write(f)
        out.release()
        save_event.clear()

threading.Thread(target=background_saver, daemon=True).start()

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("LIVE VIEW ACTIVE: Press 'q' to quit.")

frame_count = 0
try:
    while True:
        t0 = time.time()
        
        # 1. Capture
        frame = picam2.capture_array()
        video_buffer[buffer_index] = frame
        buffer_index = (buffer_index + 1) % BUFFER_SIZE
        
        # 2. Pre-proc
        blob = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        # 3. Inference
        outputs = session.run(None, {input_name: blob})
        
        # 4. Trigger Logic
        is_pothole = np.max(outputs[0][0][:, 4]) > CONF_THRESHOLD
        if is_pothole and not save_event.is_set():
            latest_detections = outputs
            save_event.set()

        # 5. LOW-IMPACT LIVE VIEW
        frame_count += 1
        if frame_count % DISPLAY_REDUCE == 0:
            # Draw a simple dot if anomaly detected so you know it's working
            display_frame = cv2.resize(frame, (320, 240)) # Smaller window = Faster
            if is_pothole:
                cv2.circle(display_frame, (20, 20), 10, (0, 0, 255), -1)
            
            cv2.putText(display_frame, f"FPS: {1.0/(time.time()-t0):.1f}", (10, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Dashcam Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"FPS: {1.0/(time.time()-t0):.2f}", end="\r")

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    cv2.destroyAllWindows()

