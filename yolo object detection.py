import torch
import numpy as np
import time
import cv2

# Load the YOLOv5 model
print("[INFO] loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()

# Initialize the video stream, allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)  # 0 is the default for the primary camera
time.sleep(2.0)

def preprocess_image(frame):
    img_resized = cv2.resize(frame, (640, 640))
    img_resized = img_resized / 255.0  # Normalize
    img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
    img_resized = np.expand_dims(img_resized, 0)  # Add batch dimension
    img_resized = torch.tensor(img_resized, dtype=torch.float32)
    return img_resized

def post_process(detections, frame):
    # Bounding boxes are in detections[0][:, :4]
    # Confidence scores are in detections[0][:, 4]
    # Class IDs are in detections[0][:, 5]
    boxes = detections[:, :4].cpu().numpy()  # Bounding boxes
    scores = detections[:, 4].cpu().numpy()  # Confidence scores
    class_ids = detections[:, 5].cpu().numpy()  # Class IDs

    for i, box in enumerate(boxes):
        if scores[i] >= 0.2:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(class_ids[i])]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {scores[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# Initialize FPS calculation
start_time = time.time()
frame_count = 0

# Loop over the frames from the video stream
while True:
    # Grab the frame from the video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Resize the frame to have a maximum width of 400 pixels
    frame = cv2.resize(frame, (400, 400))

    # Preprocess the frame
    img_preprocessed = preprocess_image(frame)
    
    # Perform object detection
    with torch.no_grad():  # Disable gradient calculation for inference
        detections = model(img_preprocessed)[0]

    # Post-process detections
    frame_with_boxes = post_process(detections, frame)
    
    # Show the output frame
    cv2.imshow("Frame", frame_with_boxes)
    key = cv2.waitKey(1) & 0xFF

    # If the "q" key was pressed, break from the loop
    if key == ord("q"):
        break

    # Update the FPS calculation
    frame_count += 1

# Calculate and display FPS
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print("[INFO] elapsed time: {:.2f}".format(elapsed_time))
print("[INFO] approx. FPS: {:.2f}".format(fps))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
