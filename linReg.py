# load needed libraries.
import os
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# sklearn utilities
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import glob

# Functions

############################


IMG_THRESHHOLD = 170

# Loads images, converts to numpy array, and displays
image_files = glob.glob(os.path.join("samples", '*.png'))
images = numpy.array([plt.imread(img) for img in image_files])
print(images.shape)
plt.imshow(images[0])
plt.show()


# Processing the images

#grayscale versions of images
new_img = []
#list of names of each image
names = []
#image width and height
imagew = [
    [30,55],
    [50, 75],
    [70, 95],
    [90, 115],
    [110, 135]
    ]

img_num = 0

#Converts images to greyscale
for img in range(len(image_files)):
    temp = image_files[img][8:13]
    for a in range(5):
        new_img.append([])
        names.append(temp[a])
        for w in range(imagew[a][0], imagew[a][1]):
            for h in range(images.shape[1]):
                if images[img][h][w][0] < IMG_THRESHHOLD/255:
                    new_img[img_num].append(images[img][h][w][0])
                else:
                    new_img[img_num].append(1.0)

        img_num += 1
        print("Image ", img_num, "/", len(images)*5, " successfully converted to greyscale")
    # img_num += 1

new_img = numpy.array(new_img)
print(new_img.shape)

# For outputting some of new_image's characters
fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # Adjust figsize as needed


# Loop through the images and corresponding axes to display them
ind = 0
for i, ax in enumerate(axes.flat):
    for x in range(5):
        # Calculate the start and end index for the current section
        start_idx = x * 50 * (imagew[x][1] - imagew[x][0])
        end_idx = (x + 1) * 50 * (imagew[x][1] - imagew[x][0])
        # Reshape only the appropriate section of new_img[i]
        reshaped_image = np.reshape(new_img[i], (imagew[x][1] - imagew[x][0], 50)).T
        ax.imshow(reshaped_image, cmap='gray', interpolation="nearest")  # Display each image section in grayscale
        ax.set_title(names[i])

# Display the grid of images
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()


# train_images, test_images, train_labels, test_labels = train_test_split()


x_train, x_test, y_train, y_test = train_test_split(new_img, names, test_size = 0.25, random_state = 0)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Training x:", x_train.shape,"y:", y_train.shape)
print("Testing x:", x_test.shape,"y:", y_test.shape)

model = linear_model.LinearRegression(fit_intercept = False)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

max_features_range = min(52, x_train.shape[1])  # Limit max_features to a reasonable number

# Each graph
scores = []
# Iterate over a valid range of features
for i in range(1, max_features_range + 1):
    # Select top i features (simulating feature selection)
    x_train_subset = x_train[:, :i]
    x_test_subset = x_test[:, :i]

    # Train the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(x_train_subset, y_train)

    # Evaluate the model
    score = model.score(x_test_subset, y_test)
    scores.append(score)

# Label and plot
plt.plot(range(1, max_features_range + 1), scores, label='Linear Regression Scores')
plt.xlabel('Number of Features')
plt.ylabel('R² Score')
plt.title('Linear Regression Performance by Features')
plt.legend()
plt.show()