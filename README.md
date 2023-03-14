Christopher M Coballes

R&D Engineer 

Hi-Techno Barrio

C/C++ and Python coding






2 Cameras

In this updated version of the program, we have added a few lines of code to calculate the distance of the object detected in the left and right camera images. We use the stereo vision output, i.e., the disparity map, to calculate the distance of the object. We have also added the camera parameters such as focal length, sensor width, and baseline to calculate the distance. Finally, we display the calculated distance of the object on both the left and right camera images.

Note: The distance calculation is an approximation and may not be very accurate due to several factors such as camera calibration, object size, and lighting conditions. It is always recommended to calibrate your camera and adjust the parameters according to your specific needs for better accuracy.


3 Cameras

The key differences between the previous implementation and this one are:

1. Three cameras are used instead of two.
2. Three stereo block matching algorithms are used to compute the disparity maps between each pair of cameras.
3. The depth of each point on the detected lines is calculated using the disparity maps and the camera parameters.
4. The 3D coordinates of the point of intersection of the three lines are calculated using linear algebra.
5. The lines and the point of intersection are drawn on all camera images.
6. All camera images are displayed.

4 Cameras

This implementation uses the same stereo block matching algorithm and line scanning approach as before, but with the addition of two more cameras. The program also calculates the 3D coordinates of the point of intersection of the four lines detected by the cameras, using a least-squares solution to the linear equation Ax = B, where A is a matrix containing the camera positions and B is a vector containing a point in the world space. Finally, the program draws the point of intersection on all four camera images and displays them in separate windows.


