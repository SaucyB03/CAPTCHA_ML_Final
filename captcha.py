# load needed libraries.
import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# sklearn utilities
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# sklearn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#import tensorflow as tf

import glob


IMG_THRESHHOLD = 170

# Loads images, converts to numpy array, and displays
image_files = glob.glob(os.path.join("samples", '*.png'))
images = numpy.array([plt.imread(img) for img in image_files])
print(images.shape)
plt.imshow(images[0])
plt.show()

new_img = []
names = []

imagew = [30,55]

for img in range(len(image_files)):
    temp = image_files[img][8:13]
    names.append(temp[0])
    new_img.append([])
    for h in range(images.shape[1]):
        for w in range(imagew[0], imagew[1]):
            if images[img][h][w][0] < IMG_THRESHHOLD/255:
                new_img[img].append(images[img][h][w][0])
            else:
                new_img[img].append(1.0)
    print("Image ", img+1, "/", len(images), " successfully converted to greyscale")

new_img = numpy.array(new_img)
print(new_img.shape)

# For outputting some of new_image's characters
fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # Adjust figsize as needed


# Loop through the images and corresponding axes to display them
for i, ax in enumerate(axes.flat):
    reshaped_image = np.reshape(new_img[i], (50, imagew[1] - imagew[0]))
    ax.imshow(reshaped_image, cmap='gray', interpolation="nearest")  # Display each image in grayscale

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

# scores = []
# for i in range(1, x_train.shape[0]):
#     #add your code here.
#     treeI = DecisionTreeClassifier(max_features = i)
#     treeI.fit(x_train, y_train)
#     scores.append(treeI.score(x_test, y_test))
#
# plt.plot(scores)

tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)
print(tree.score(x_test, y_test))

forest = RandomForestClassifier()
forest.fit(x_train, y_train)
print(forest.score(x_test,y_test))