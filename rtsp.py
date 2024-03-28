from ultralytics import YOLO
import subprocess as sp
import numpy as np
import threading
import time
import cv2

# Load a model
# model = YOLO("yolov8n.pt") # Tiny model
model = YOLO("yolov8s.pt") # Small model
# model = YOLO("yolov8m.pt") # Medium model

# Constants
THRESHOLD = 0.5
WIDTH = 1920
HEIGHT = 1080
FPS = 20

# RTSP input stream
RTSP_STREAM = 'rtsp://admin:admin@169.226.53.111:554/live'

# FFmpeg command to convert H.264 to raw video
ffmpeg_command = [
    'ffmpeg',
    '-timeout', '10000000',
    '-rtsp_transport', 'udp',
    '-i', RTSP_STREAM,
    '-reconnect_at_eof', '1',
    '-reconnect_streamed', '1',
    '-f', 'rawvideo',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-pix_fmt', 'bgr24',
    '-',
]

last_frame = None
stop_event = threading.Event()

def update_last_frame(ffmpeg_process):
    global last_frame
    while not stop_event.is_set():
        # Read a frame from the FFmpeg process
        raw_frame = ffmpeg_process.stdout.read(WIDTH * HEIGHT * 3)  # Adjust the frame size accordingly
        if not raw_frame:
            continue
        else:
            # Update the last frame by converting the raw frame to an OpenCV image
            last_frame = cv2.resize(np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)), (WIDTH, HEIGHT))
            del raw_frame


if __name__ == '__main__':
    # Start FFmpeg process
    ffmpeg_process = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE)

    # Create a thread for capturing the last frame from the camera
    frame_thread = threading.Thread(target=update_last_frame, args=(ffmpeg_process,))
    frame_thread.start()

    # used to record the time when we processed last frame
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        if last_frame is None:
            continue

        return_code = ffmpeg_process.poll()
        if return_code is not None:
            print("FFmpeg connection lost.")
            stop_event.set()
            break
        
        frame = last_frame.copy()

        # Visualize the results on the frame
        results = model(frame, device='mps', verbose=False, conf=THRESHOLD)
        annotated_frame = results[0].plot()

        # Calculating the fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        fps = str(int(fps))
        prev_frame_time = new_frame_time

        # Display the FPS
        cv2.putText(annotated_frame, fps, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('YOLOv8', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    
    # Release resources
    cv2.destroyAllWindows()
    frame_thread.join()
    ffmpeg_process.terminate()






