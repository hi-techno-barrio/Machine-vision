import cv2
import numpy as np

# Load YOLO object detection model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set classes and colors
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Define function for object detection
def detect_objects(image):
    # Get image dimensions
    height, width, channels = image.shape

    # Create blob from input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set input to YOLO model
    net.setInput(blob)

    # Get output from YOLO model
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize variables for object detection
    boxes = []
    confidences = []
    class_ids = []

    # Loop through each output from YOLO model
    for output in outputs:
        # Loop through each detection
        for detection in output:
            # Get confidence, class ID, and bounding box coordinates
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box, confidence, and class ID to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize variables for object names and distances
    object_names = []
    distances = []

    # Check if there is at least one detection
    if len(indices) > 0:
        # Loop through each detection after non-maximum suppression
        for i in indices.flatten():
            # Get class name and confidence for current detection
            class_name = classes[class_ids[i]]
            confidence = confidences[i]

            # Compute distance to current detection
            # Assumes known object size and focal length of camera
            object_size = 0.2 # meters
            focal_length = 500 # pixels
            distance = object_size * focal_length / boxes[i][2]

            # Add object name and distance to lists
            object_names.append(class_name)
            distances.append(distance)

            # Draw bounding box and label on image
            color = colors[class_ids[i]]
            cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color, 2)
            text = f"{class_name}: {distance:.2f}m"
            cv2.putText(image, text, (boxes[i][0], boxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return object_names, distances, image

# Initialize video capture for each camera
cap_left = cv2.VideoCapture
        # Draw bounding box and label for each object detected
        for detection in detections:
            class_id = detection[0]
            confidence = detection[1]
            if class_id in self.classes:
                class_name = self.classes[class_id]
                label = '{}: {:.2f}'.format(class_name, confidence)
                color = self.colors[class_id]
                x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
                xmin, ymin, xmax, ymax = self.get_bounding_box(x, y, w, h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate distance to object
                distance = self.calculate_distance_to_object(w)
                
                # Determine which camera detected the object
                if camera_id == 0:
                    print("Object detected in front camera:", class_name, "distance:", distance)
                elif camera_id == 1:
                    print("Object detected in back camera:", class_name, "distance:", distance)
                elif camera_id == 2:
                    print("Object detected in left camera:", class_name, "distance:", distance)
                elif camera_id == 3:
                    print("Object detected in right camera:", class_name, "distance:", distance)
                    
        return frame
