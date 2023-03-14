import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('image.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
thresh12 = cv2.threshold(disparity12, 0, 255, cv2.THRESH_BINARY)[1]
thresh13 = cv2.threshold(disparity13, 0, 255, cv2.THRESH_BINARY)[1]
thresh14 = cv2.threshold(disparity14, 0, 255, cv2.THRESH_BINARY)[1]
thresh23 = cv2.threshold(disparity23, 0, 255, cv2.THRESH_BINARY)[1]
thresh24 = cv2.threshold(disparity24, 0, 255, cv2.THRESH_BINARY)[1]
thresh34 = cv2.threshold(disparity34, 0, 255, cv2.THRESH_BINARY)[1]

# apply morphological opening to remove noise
kernel = np.ones((3, 3), np.uint8)
opening12 = cv2.morphologyEx(thresh12, cv2.MORPH_OPEN, kernel)
opening13 = cv2.morphologyEx(thresh13, cv2.MORPH_OPEN, kernel)
opening14 = cv2.morphologyEx(thresh14, cv2.MORPH_OPEN, kernel)
opening23 = cv2.morphologyEx(thresh23, cv2.MORPH_OPEN, kernel)
opening24 = cv2.morphologyEx(thresh24, cv2.MORPH_OPEN, kernel)
opening34 = cv2.morphologyEx(thresh34, cv2.MORPH_OPEN, kernel)

# scan the horizon line for lines in all camera pairs
for i in range(gray1.shape[1] // step):
    x1, y1 = i * step, horizon
    x2, y2 = (i + 1) * step, horizon

    if opening12 is not None:
        for line in cv2.HoughLinesP(opening12[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth12 = (focal_length * baseline12) / (disparity12[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame1, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    if opening13 is not None:
        for line in cv2.HoughLinesP(opening13[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth13 = (focal_length * baseline13) / (disparity13[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame1, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    if opening14 is not None:
        for line in cv2.HoughLinesP(opening14[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth14 = (focal_length * baseline14) / (disparity14[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame1, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    if opening23 is not None:
        for line in cv2.HoughLinesP(opening23[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth23 = (focal_length * baseline23) / (disparity23[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame2, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    if opening24 is not None:
        for line in cv2.HoughLinesP(opening24[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth24 = (focal_length * baseline24) / (disparity24[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame2, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    if opening34 is not None:
        for line in cv2.HoughLinesP(opening34[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
            x1_, y1_, x2_, y2_ = line[0]
            depth34 = (focal_length * baseline34) / (disparity34[y1 + y1_, x1 + x1_] + 0.0000001)
            cv2.line(frame3, (x1 + x1_, y1 + y1_), (x1 + x2_, y1 + y2_), (0, 255, 0), 2)

    # object detection on all frames
    blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)
    class_ids1 = []
    confidences1 = []
    boxes1 = []
    for out in outs1:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * gray1.shape[1])
                center_y = int(detection[1] * gray1.shape[0])
                w = int(detection[2] * gray1.shape[1])
                h = int(detection[3] * gray1.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids1.append(class_id)
                confidences1.append(float(confidence))
                boxes1.append([x, y, w, h])
    indices1 = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.5, 0.4)

    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob2)
outs2 = net.forward(output_layers)
class_ids2 = []
confidences2 = []
boxes2 = []
for out in outs2:
for detection in out:
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence > 0.5:
center_x = int(detection[0] * gray2.shape[1])
center_y = int(detection[1] * gray2.shape[0])
w = int(detection[2] * gray2.shape[1])
h = int(detection[3] * gray2.shape[0])
x = int(center_x - w / 2)
y = int(center_y - h / 2)
class_ids2.append(class_id)
confidences2.append(float(confidence))
boxes2.append([x, y, w, h])
indices2 = cv2.dnn.NMSBoxes(boxes2, confidences2, 0.5, 0.4)    blob3 = cv2.dnn.blobFromImage(frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob3)
    outs3 = net.forward(output_layers)
    class_ids3 = []
    confidences3 = []
    boxes3 = []
    for out in outs3:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * gray3.shape[1])
                center_y = int(detection[1] * gray3.shape[0])
                w = int(detection[2] * gray3.shape[1])
                h = int(detection[3] * gray3.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids3.append(class_id)
                confidences3.append(float(confidence))
                boxes3.append([x, y, w, h])
    indices3 = cv2.dnn.NMSBoxes(boxes3, confidences3, 0.5, 0.4)

    blob4 = cv2.dnn.blobFromImage(frame4, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob4)
    outs4 = net.forward(output_layers)
    class_ids4 = []
    confidences4 = []
    boxes4 = []
    for out in outs4:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * gray4.shape[1])
                center_y = int(detection[1] * gray4.shape[0])
                w = int(detection[2] * gray4.shape[1])
                h = int(detection[3] * gray4.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids4.append(class_id)
                confidences4.append(float(confidence))
                boxes4.append([x, y, w, h])
    indices4 = cv2.dnn.N
MSBoxes(boxes4, confidences4, 0.5, 0.4)    # draw bounding boxes and labels on all frames
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    if len(indices1) > 0:
        for i in indices1.flatten():
            x, y, w, h = boxes1[i]
            label = str(class_names[class_ids1[i]])
            confidence = str(round(confidences1[i], 2))
            color = colors[class_ids1[i]]
            cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame1, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    if len(indices2) > 0:
        for i in indices2.flatten():
            x, y, w, h = boxes2[i]
            label = str(class_names[class_ids2[i]])
            confidence = str(round(confidences2[i], 2))
            color = colors[class_ids2[i]]
            cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame2, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    if len(indices3) > 0:
        for i in indices3.flatten():
            x, y, w, h = boxes3[i]
            label = str(class_names[class_ids3[i]])
            confidence = str(round(confidences3[i], 2))
            color = colors[class_ids3[i]]
            cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame3, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    if len(indices4) > 0:
        for i in indices4.flatten():
            x, y, w, h = boxes4[i]
            label = str(class_names[class_ids4[i]])
            confidence = str(round(confidences4[i], 2))
            color = colors[class_ids4[i]]
            cv2.rectangle(frame4, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame4, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    # show all frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    cv2.imshow('Camera 3', frame3)
    cv2.imshow('Camera 4', frame4)

    # break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the resources
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
if name == 'main':
main()
