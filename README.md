Christopher M Coballes

R&D Engineer 

Hi-Techno Barrio

C/C++ and Python coding









3 Cameras

The key differences between the previous implementation and this one are:

1. Three cameras are used instead of two.
2. Three stereo block matching algorithms are used to compute the disparity maps between each pair of cameras.
3. The depth of each point on the detected lines is calculated using the disparity maps and the camera parameters.
4. The 3D coordinates of the point of intersection of the three lines are calculated using linear algebra.
5. The lines and the point of intersection are drawn on all camera images.
6. All camera images are displayed.


