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

chosen_chars = ['2','d','g']

img_num = 0

#Converts images to greyscale
for img in range(len(image_files)):
    temp = image_files[img][8:13]

    for a in range(5):
        new_img.append([])
        if temp[a] in chosen_chars:
            names.append(True)
        else:
            names.append(False)
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
for i, ax in enumerate(axes.flat):

    for x in range(5):
        # Calculate the start and end index for the current section
        start_idx = x * 50 * (imagew[x][1] - imagew[x][0])
        end_idx = (x + 1) * 50 * (imagew[x][1] - imagew[x][0])
        # Reshape only the appropriate section of new_img[i]
        reshaped_image = np.reshape(new_img[i], (imagew[x][1] - imagew[x][0], 50)).T
        ax.imshow(reshaped_image, cmap='gray', interpolation="nearest")  # Display each image section in grayscale

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