import cv2
import time
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_frame_time = 0
new_frame_time = 0

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, stream=True, verbose=False, vid_stride=5, conf=0.4)
        people_count = 0

        for r in results:
            for box in r.boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                text = str(classNames[int(box.cls[0])]) + ": {:.2f}".format(box.conf[0])
                if int(box.cls[0]) == 0:
                    people_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 184, 0), 3)
                cv2.putText(frame, text, [x1, y1-5], cv2.FONT_HERSHEY_SIMPLEX, 1, (172, 60, 233), 2)

  
        # Calculating the fps
        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time

        cv2.putText(frame, str(fps), (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'People Count:{people_count}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8", frame)
        

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # if fps == 0:
        #     break
    else:
        break

cap.release()
cv2.destroyAllWindows()