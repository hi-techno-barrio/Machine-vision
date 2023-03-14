import cv2
import numpy as np

# initialize camera capture
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# set the resolution for both cameras
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# initialize stereo block matching algorithm
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# camera parameters for distance calculation
focal_length = 3.6  # in mm
sensor_width = 6.17  # in mm
baseline = 75  # in mm

while True:
    # read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # compute disparity map
    disparity = stereo.compute(gray1, gray2)

    # apply thresholding to disparity map
    thresh = cv2.threshold(disparity, 0, 255, cv2.THRESH_BINARY)[1]

    # apply morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # detect lines in left and right images
    lines1 = cv2.HoughLinesP(opening[:, :320], 1, np.pi / 180, 100, None, 50, 10)
    lines2 = cv2.HoughLinesP(opening[:, 320:], 1, np.pi / 180, 100, None, 50, 10)

    # draw lines on both images and calculate distance
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # calculate distance of object
            depth = (focal_length * baseline) / (disparity[y1, x1] + 0.0000001)
            cv2.putText(frame1, f"Distance: {depth:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame2, (x1 + 320, y1), (x2 + 320, y2), (0, 255, 0), 2)
            # calculate distance of object
            depth = (focal_length * baseline) / (disparity[y1, x1 + 320] + 0.0000001)
            cv2.putText(frame2, f"Distance: {depth:.2f} cm", (x1 + 320, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0
        ), 2)

# display images
cv2.imshow('Left Camera', frame1)
cv2.imshow('Right Camera', frame2)

# exit on ESC key
if cv2.waitKey(1) == 27:
    break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
