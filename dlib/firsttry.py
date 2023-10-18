import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the facial landmark predictor from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"  # You'll need to download this model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load an image
image_path = "/Users/ariellastefansky/python1/images.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale for facial feature detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)


for face in faces:
    # Get the facial landmarks
    landmarks = predictor(gray, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]  # Store all landmarks as (x, y) tuples

    # Create the images
    dots_image = image.copy()
    lines_image1 = image.copy()
    lines_image = np.zeros_like(image)

    # Draw dots on the images
    for i in range(68):  # 68 landmarks
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(dots_image, (x, y), 3, (0, 0, 255), -1)  # Draw a red dot
    


    left_eyebrow_points = []
    right_eyebrow_points = []

    for i in range(17, 22):  # Landmarks 17 to 21 correspond to the left eyebrow
        x, y = landmarks.part(i).x, landmarks.part(i).y
        left_eyebrow_points.append((x, y))
    for i in range(22, 27):  # Landmarks 22 to 26 correspond to the right eyebrow
        x, y = landmarks.part(i).x, landmarks.part(i).y
        right_eyebrow_points.append((x, y))

    # Connect the dots to draw a continuous line around the face
    for i in range(0, 16):
        cv2.line(lines_image1, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image1, points[16], points[0], (0, 255, 0), 1)

    # Connect the dots to draw continuous lines around the eyes
    for i in range(36, 41):
        cv2.line(lines_image1, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image1, points[41], points[36], (0, 255, 0), 1)

    for i in range(42, 47):
        cv2.line(lines_image1, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image1, points[47], points[42], (0, 255, 0), 1)

    # Connect the dots to draw a continuous line for the nose
    for i in range(27, 35):
        cv2.line(lines_image1, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image1, points[35], points[27], (0, 255, 0), 1)

    # Connect the dots to draw a continuous line for the mouth
    for i in range(48, 59):
        cv2.line(lines_image1, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image1, points[59], points[48], (0, 255, 0), 1)
    cv2.line(lines_image1, points[60], points[48], (0, 255, 0), 1)
    cv2.line(lines_image1, points[60], points[61], (0, 255, 0), 1)
    cv2.line(lines_image1, points[61], points[62], (0, 255, 0), 1)
    cv2.line(lines_image1, points[62], points[63], (0, 255, 0), 1)
    cv2.line(lines_image1, points[63], points[64], (0, 255, 0), 1)
    cv2.line(lines_image1, points[64], points[65], (0, 255, 0), 1)
    cv2.line(lines_image1, points[65], points[66], (0, 255, 0), 1)
    cv2.line(lines_image1, points[66], points[67], (0, 255, 0), 1)
    cv2.line(lines_image1, points[67], points[48], (0, 255, 0), 1)

    # Connect the left eyebrow points with lines
    for i in range(len(left_eyebrow_points) - 1):
        cv2.line(lines_image1, left_eyebrow_points[i], left_eyebrow_points[i + 1], (0, 255, 0), 1)

    # Connect the right eyebrow points with lines
    for i in range(len(right_eyebrow_points) - 1):
        cv2.line(lines_image1, right_eyebrow_points[i], right_eyebrow_points[i + 1], (0, 255, 0), 1)



    # Connect the dots to draw a continuous line around the face
    for i in range(0, 16):
        cv2.line(lines_image, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image, points[16], points[0], (0, 255, 0), 1)

    # Connect the dots to draw continuous lines around the eyes
    for i in range(36, 41):
        cv2.line(lines_image, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image, points[41], points[36], (0, 255, 0), 1)

    for i in range(42, 47):
        cv2.line(lines_image, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image, points[47], points[42], (0, 255, 0), 1)

    # Connect the dots to draw a continuous line for the nose
    for i in range(27, 35):
        cv2.line(lines_image, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image, points[35], points[27], (0, 255, 0), 1)

    # Connect the dots to draw a continuous line for the mouth
    for i in range(48, 59):
        cv2.line(lines_image, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(lines_image, points[59], points[48], (0, 255, 0), 1)
    cv2.line(lines_image, points[60], points[48], (0, 255, 0), 1)
    cv2.line(lines_image, points[60], points[61], (0, 255, 0), 1)
    cv2.line(lines_image, points[61], points[62], (0, 255, 0), 1)
    cv2.line(lines_image, points[62], points[63], (0, 255, 0), 1)
    cv2.line(lines_image, points[63], points[64], (0, 255, 0), 1)
    cv2.line(lines_image, points[64], points[65], (0, 255, 0), 1)
    cv2.line(lines_image, points[65], points[66], (0, 255, 0), 1)
    cv2.line(lines_image, points[66], points[67], (0, 255, 0), 1)
    cv2.line(lines_image, points[67], points[48], (0, 255, 0), 1)

    # Connect the left eyebrow points with lines
    for i in range(len(left_eyebrow_points) - 1):
        cv2.line(lines_image, left_eyebrow_points[i], left_eyebrow_points[i + 1], (0, 255, 0), 1)

    # Connect the right eyebrow points with lines
    for i in range(len(right_eyebrow_points) - 1):
        cv2.line(lines_image, right_eyebrow_points[i], right_eyebrow_points[i + 1], (0, 255, 0), 1)


# Create a subplot for displaying the images
plt.figure(figsize=(12, 4))

# Original Image with Dots
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Original Image with Dots
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(dots_image, cv2.COLOR_BGR2RGB))
plt.title("Dots Image")
plt.axis('off')

# Image with Facial Features Map
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(lines_image1, cv2.COLOR_BGR2RGB))
plt.title("Facial Features Map")
plt.axis('off')

# Facial Features Map Only
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB))
plt.title("Facial Features Map Only")
plt.axis('off')

plt.tight_layout()
plt.show()

