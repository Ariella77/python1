import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the images
image_paths = ["/Users/ariellastefansky/python1/images.jpg", "/Users/ariellastefansky/python1/licensed-image.jpg"]
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


# Define the desired dimensions
desired_width = 1350
desired_height = 1380

# Initialize lists to store the cropped faces
cropped_faces = []

# Crop the faces in the images
for img in images:
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        # Resize the cropped face to the desired dimensions
        resized_face = cv2.resize(face, (desired_width, desired_height))
        cropped_faces.append(resized_face.flatten())

# Stack the cropped faces together into one dataset
face_data = np.stack(cropped_faces, axis=0)

n_components = len(images)*len(images) # Number of eigenfaces to extract
pca = PCA(n_components=n_components)

# Fit PCA to the cropped face data
pca.fit(face_data)

# Get the eigenfaces
eigenfaces = pca.components_

# Visualize the eigenfaces
plt.figure(figsize=(12, 6))
for i in range(n_components):
    eigenface = eigenfaces[i]
    plt.subplot(1, n_components, i + 1)
    eigenface_rescaled = 255 * (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
    plt.imshow(eigenface_rescaled.reshape(desired_height, desired_width), cmap='gray')
    plt.title(f'Eigenface {i + 1}')
    plt.axis('off')

plt.show()