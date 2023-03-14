import cv2
import numpy as np

# initialize camera capture
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)
cap4 = cv2.VideoCapture(3)

# set the resolution for all cameras
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# initialize stereo block matching algorithms
stereo12 = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo13 = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo14 = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo23 = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo24 = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo34 = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# camera parameters for distance calculation
focal_length = 3.6  # in mm
sensor_width = 6.17  # in mm
baseline12 = 75  # in mm
baseline13 = 100  # in mm
baseline14 = 50  # in mm
baseline23 = 75  # in mm
baseline24 = 50  # in mm
baseline34 = 75  # in mm

# line scanning parameters
horizon = 200  # horizon line for line scanning
step = 10  # pixel step size for line scanning

while True:
    # read frames from all cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)

    # compute disparity maps
    disparity12 = stereo12.compute(gray1, gray2)
    disparity13 = stereo13.compute(gray1, gray3)
    disparity14 = stereo14.compute(gray1, gray4)
    disparity23 = stereo23.compute(gray2, gray3)
    disparity24 = stereo24.compute(gray2, gray4)
    disparity34 = stereo34.compute(gray3, gray4)

    # apply thresholding to disparity maps
    thresh12 = cv2.threshold(disparity12, 0, 255, cv2.THRESH_BINARY)[1]
    thresh13 = cv2.threshold(disparity13, 0, 255, cv2.THRESH_BINARY)[1]
    thresh14 = cv2.threshold(disparity14,0, 255, cv2.THRESH_BINARY)[1]
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
        for line in cv2.HoughLines P(opening23[y1:y2, :], 1, np.pi / 180, 100, None, 50, 10):
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

# calculate 3D coordinates of the point of intersection of the four lines
a = np.array([(0, 0, depth12), (gray1.shape[1], 0, depth14),
              (gray2.shape[1], 0, depth24), (gray3.shape[1], 0, depth34)])
A = np.dot(a.T, a)
B = np.array([0, 0, 0, 1])
X = np.linalg.lstsq(A, B, rcond=None)[0]
x, y, z = X[:3]

# draw point of intersection on all camera images
cv2.circle(frame1, (int(x), int(y)), 5, (255, 0, 0), -1)
cv2.circle(frame2, (int(x), int(y)), 5, (255, 0, 0), -1)
cv2.circle(frame3, (int(x), int(y)), 5, (255, 0, 0), -1)
cv2.circle(frame4, (int(x), int(y)), 5, (255, 0, 0), -1)

# display images
cv2.imshow('Camera 1', frame1)
cv2.imshow('Camera 2', frame2)
cv2.imshow('Camera 3', frame3)
cv2.imshow('Camera 4', frame4)

# exit on ESC key
if cv2.waitKey(1) == 27:
    break
     cap1.release()
     cap2.release()
     cap3.release()
     cap4.release()
cv2.destroyAllWindows()



    
    
