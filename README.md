# 6DoF-Audio-Toolbox
This repository contains my utility functions designed to assist with various spatial audio tasks, including sound localization, 3D audio rendering, and spatial audio analysis.

## **detect_micarray_rotation_by_image.py**  
Zylia microphone array rotational mismatch calculations via top view photograph of a recording layout.

Example input image of 5 Zylia Mics rotated arbitrarily:   
<img src="images/5_zylias.png" width="500" style="display: block; margin-left: 0;"/>

### Features:
- Detects circular objects in a grayscale image using the Hough Circle Transform.
- Sorts detected circles from upper-left to bottom-right (0 being the reference)
- Crops detected circles individually with a margin (not to accidentally crop the image) for feature detection.
- Calculates the rotational misalignment between the reference (upper-leftmost object) and all others using ORB feature matching and affine transformation.
- Visualizes detected circles and feature matches for verification.

### Usage:
1. Place the top view image of the recording setup in the same directory as the script or update the `new_image_path` variable.
2. Run the script to detect and visualize the circles and their rotational misalignment.

### Dependencies:
- OpenCV
- NumPy
- Matplotlib

Detected circles for the example image:   
<img src="images/5_zylias_detected.png" width="500" style="display: block; margin: 0;"/>

Features found for the object 1 and 4:  
<img src="images/eg_feature_detected.png" width="500" style="display: block; margin: 0;"/>

Console output:
Rotation Angles: [  0.  -16.3   5.5 -90.   45.4]

## convertangle.py  
Angle/coordinate transformations, mic. array location rotations

## sputil.py  
Spatial audio processing utility functions such as ambix-shd-ambisonics conversions, wav-shd conversions, time-frequency transforms, for multichannel audio.
