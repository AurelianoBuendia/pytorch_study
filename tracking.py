
import os
import cv2
from ultralytics import YOLO


model = YOLO("./models/yolo11n.pt")
videos_folder = "./videos"
output_folder = "./output"
input_video_file = "84196_960x540.mp4"
output_video_file = "84196_960x540_tracking.mp4"
input_video_path = os.path.join(videos_folder, input_video_file)
output_video_path = os.path.join(output_folder, output_video_file)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Could not open video")
    exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Set the line position to 60% of the screen width
center_line_x = int(width * 0.6)

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# In tracked_objects, keep each object's "last x-coordinate" and "crossed flag"
tracked_objects = {}
# Total count of objects that have crossed the line
cross_count = 0
# Count per class for crossed objects(e.g, {'person'=0, 'car'=3, ...})
class_cross_count = {}

# Perform tracking on the entire video (with stream=True to retrieve each frame result sequentially)
results = model.track(source=input_video_path, show=False, stream=True)

for result in results:
    frame = result.orig_img # Original frame image
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # Get coordinates in format xyxy
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2
            if hasattr(box, "id") and box.id is not None:
                obj_id = int(box.id)
            else:
                continue
            # Retrieve class ID and class name
            if hasattr(box, "cls") and box.cls is not None:
                class_id = int(box.cls)
                class_name = model.names[class_id]
            else:
                class_name = "Unknown"
            # If the object already exists in tracked_objects, compare the previous x_coordinate
            if obj_id in tracked_objects:
                prev_x = tracked_objects[obj_id]['last_x']
                # If it has not been counted for crossing, perform crossing check
                if not tracked_objects[obj_id]['counted']:
                    if (prev_x < center_line_x and cx >= center_line_x) or \
                        (prev_x > center_line_x and cx <= center_line_x):
                        cross_count += 1
                        tracked_objects[obj_id]['counted'] = True
                        # Update count per class
                        if class_name in class_cross_count:
                            class_cross_count[class_name] += 1
                        else:
                            class_cross_count[class_name] = 1
                # Update the x-coordinate
                tracked_objects[obj_id]['last_x'] = cx
            else:
                # Register new object information
                tracked_objects[obj_id] = {'last_x': cx, 'counted': False}
            # Draw the bounding box, ID and class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"ID:{obj_id} {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Draw the vertical line
    cv2.line(frame, (center_line_x, 0), (center_line_x, height), (0, 0, 255), 2)
    # Display the crossing count (total and per class breakdown)
    text_total = f"Total count: {cross_count}"
    cv2.putText(frame, text_total, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
    # Display the count per class at the top
    y_offset = 60
    for cls, count in class_cross_count.items():
        text_cls = f"{cls}: {count}"
        cv2.putText(frame, text_cls, (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
    out.write(frame)
out.release()
print("Processing complete. The results are saved in ", output_video_file)

