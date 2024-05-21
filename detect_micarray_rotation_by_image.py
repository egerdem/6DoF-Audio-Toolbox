import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate rotation angle between two images
def calculate_rotation_angle(img1, img2):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the rotation matrix
    M, mask = cv2.estimateAffinePartial2D(pts2, pts1)

    # Extract the rotation angle from the rotation matrix
    angle = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
    return angle, matches, kp1, kp2

def sort_by_rows(circles, row_height=100):
    # Sort detected objects from upper-left to bottom-right
    # Sort primarily by y-coordinate divided by row_height to group by rows, then by x-coordinate
    # circle has rows of (x_coordinates, y_coordinates, r)
    sorted_circles = sorted(circles, key=lambda c: (c[1] // row_height, c[0]))
    return sorted_circles

# Load the new image with multiple RSMAs
new_image_path = "5_zylias.png"  # Update the image path/name as needed
new_image = cv2.imread(new_image_path)

# Check if the image was loaded successfully
if new_image is None:
    print("Error: Unable to load image.")
else:
    print("Image loaded successfully. Image shape:", new_image.shape)

    # Convert image to grayscale
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Use HoughCircles to detect the spherical objects
    circles = cv2.HoughCircles(
        new_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,  # Increase minDist to ensure circles are well-separated
        param1=50,
        param2=100,  # Increase param2 to reduce false positives
        minRadius=50,
        maxRadius=100
    )

    # Ensure at least some circles were found
    if circles is not None:
        print(f" no. of detected circles: {circles.shape}")
        circles = np.round(circles[0, :]).astype("int")
        circles_list = [list(i) for i in circles]
        sorted_circles = sort_by_rows(circles_list, row_height=100)
        # Define a margin to avoid cutting off portions of the spherical object
        margin = 10

        # Crop the detected spherical objects with margin
        new_crops = []
        output = new_image.copy()

        for (x, y, r) in sorted_circles:
            crop = new_gray[max(0, y-r-margin):min(new_gray.shape[0], y+r+margin), max(0, x-r-margin):min(new_gray.shape[1], x+r+margin)]
            new_crops.append(crop)

            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # Display the detected objects
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title(f'Detected Circles: {len(sorted_circles)}')
            plt.axis('off')
            plt.show()

        # Iterate over pairs of cropped images and calculate rotation angles
        num_objects = len(new_crops)
        rotation_angles = np.zeros(num_objects)

        for j in range(num_objects):
            #calculate the angle between the first object (0) (upper-leftmost circle) and the others (j)
            angle, matches, kp1, kp2 = calculate_rotation_angle(new_crops[0], new_crops[j])
            rotation_angles[j] = np.round(angle, 1)

            # Draw the matches
            match_img = cv2.drawMatches(new_crops[0], kp1, new_crops[j], kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the matches
            plt.figure(figsize=(20, 10))
            plt.imshow(match_img)
            plt.title(f'Feature Matches between Object {1} and Object {j+1}')
            plt.axis('off')
            plt.show()

        # Print rotation angles matrix
        print(f"Rotation Angles: {rotation_angles}")
    else:
        print("No spherical objects detected.")