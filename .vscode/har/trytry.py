import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Haar Cascade classifiers for face, eye, and iris detection
def load_classifiers():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    mouth_classifier = cv2.CascadeClassifier('/Users/ariellastefansky/python1/.vscode/haarcascade_mcs_mouth.xml')
    nose_classifier = cv2.CascadeClassifier('/Users/ariellastefansky/python1/.vscode/haarcascade_mcs_nose.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    iris_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    return face_classifier, mouth_classifier, nose_classifier, eye_classifier, iris_classifier

def detect_faces(img, face_classifier):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))
    return faces

def detect_eyes_and_irises(img, faces, eye_classifier, iris_classifier):
    eye_coordinates = []  # Store coordinates of detected eyes
    iris_coordinates = []  # Store coordinates of detected irises

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        
        eyes = eye_classifier.detectMultiScale(face_region, scaleFactor=1.01, minNeighbors=5)
        
        for (ex, ey, ew, eh) in eyes:
            eye_x, eye_y = x + ex, y + ey
            eye_coordinates.append((eye_x, eye_y, ew, eh))
            
            eye_region = face_region[ey:ey+eh, ex:ex+ew]
            irises = iris_classifier.detectMultiScale(eye_region)
            
            for (ix, iy, iw, ih) in irises:
                iris_x, iris_y = eye_x + ix, eye_y + iy
                iris_coordinates.append((iris_x, iris_y, iw, ih))
    
    return eye_coordinates, iris_coordinates


# Detect mouth and nose within the bottom half of each detected face region
def detect_mouth_and_nose(img, faces, mouth_classifier, nose_classifier):
    mouth_coordinates = []  # Store coordinates of detected mouths
    nose_coordinates = []  # Store coordinates of detected noses

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        # Calculate the midpoint of the bottom half of the face region
        mid_y = y + (h // 2)

        # Define the bottom half of the face region
        bottom_half = img[mid_y:y+h, x:x+w]

        mouths = mouth_classifier.detectMultiScale(bottom_half, scaleFactor=1.1, minNeighbors=10)
        noses = nose_classifier.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=9)

        for (mx, my, mw, mh) in mouths:
            mouth_x, mouth_y = x + mx, my + mid_y + y # Corrected calculation
            mouth_coordinates.append((mouth_x, mouth_y, mw, mh))

        for (nx, ny, nw, nh) in noses:
            nose_x, nose_y = x + nx, ny + y # Corrected calculation
            nose_coordinates.append((nose_x, nose_y, nw, nh))

    return mouth_coordinates, nose_coordinates


def segment_iris(eye_img):
    # Perform color-based segmentation to detect the iris region
    # Modify these lower and upper HSV values according to your specific image
    lower_iris = np.array([30, 30, 30])
    upper_iris = np.array([100, 255, 255])
    hsv_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2HSV)
    iris_mask = cv2.inRange(hsv_img, lower_iris, upper_iris)

    return iris_mask

imagePath = '/Users/ariellastefansky/python1/kjh.jpg'
img = cv2.imread(imagePath)

face_classifier, mouth_classifier, nose_classifier, eye_classifier, iris_classifier = load_classifiers()

faces = detect_faces(img, face_classifier)
mouth_coordinates, nose_coordinates = detect_mouth_and_nose(img, faces, mouth_classifier, nose_classifier)
eye_coordinates, iris_coordinates = detect_eyes_and_irises(img, faces, eye_classifier, iris_classifier)

# Convert the image to color (BGR to RGB)
color_img_eye = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Create a copy of the original image
color_img_mouth_nose = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Create another copy for mouth/nose detection
# Specify the width and height for the blank image (use the dimensions you prefer)
width, height = img.shape[1], img.shape[0]

# Create a blank white image
color_img_whole = 255 * np.ones((height, width, 3), np.uint8)


# Draw rectangles for mouths and noses on the color image
for (mx, my, mw, mh) in mouth_coordinates:
    cv2.rectangle(color_img_mouth_nose, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

for (nx, ny, nw, nh) in nose_coordinates:
    cv2.rectangle(color_img_mouth_nose, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

# Draw rectangles for eyes and irises on the color image
for (ex, ey, ew, eh) in eye_coordinates:
    cv2.rectangle(color_img_eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

for (ix, iy, iw, ih) in iris_coordinates:
    cv2.rectangle(color_img_eye, (ix, iy), (ix + iw, iy + ih), (0, 0, 255), 2)



# Draw rectangles for mouths and noses on the color image for whole
for (x, y, w, h) in faces:
    cv2.rectangle(color_img_whole, (x, y), (x + w, y + h), (0, 255, 0), 4)

for (mx, my, mw, mh) in mouth_coordinates:
    cv2.rectangle(color_img_whole, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

for (nx, ny, nw, nh) in nose_coordinates:
    cv2.rectangle(color_img_whole, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

# Draw rectangles for eyes and irises on the color image for whole
for (ex, ey, ew, eh) in eye_coordinates:
    cv2.rectangle(color_img_whole, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

for (ix, iy, iw, ih) in iris_coordinates:
    cv2.rectangle(color_img_whole, (ix, iy), (ix + iw, iy + ih), (0, 0, 255), 2)


plt.figure(figsize=(6, 6))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 3, 2)
face_img = img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
plt.title('Face Detection')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(color_img_eye)
plt.title('Eye and Iris Detection')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(color_img_mouth_nose)
plt.title('Mouth and Nose Detection')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(color_img_whole)
plt.title('Whole Detection')
plt.axis('off')

plt.show()
