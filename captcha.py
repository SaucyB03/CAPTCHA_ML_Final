# load needed libraries.
import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn
#
# # sklearn utilities
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
#
# # sklearn models
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf

import glob


IMG_THRESHHOLD = 170


image_files = glob.glob(os.path.join("samples", '*.png'))

images = numpy.array([plt.imread(img) for img in image_files])

print(images.shape)

plt.imshow(images[0])
plt.show()

new_img = {}

for img in range(len(image_files)):
    temp = image_files[img][8:13]
    new_img[temp] = []
    for h in range(images.shape[1]):
        new_img[temp].append([])
        for w in range(25,60):
            if images[img][h][w][0] < IMG_THRESHHOLD/255:
                new_img[temp][h].append(images[img][h][w][0])
            else:
                new_img[temp][h].append(1.0)
    print("Image ", img+1, "/", len(images), " successfully converted to greyscale")
#


fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # Adjust figsize as needed

first_of_l = {}

for file, img  in new_img.items():
    if file[0] not in first_of_l:
        first_of_l[file[0]] = img

# Loop through the images and corresponding axes to display them
for i, ax in enumerate(axes.flat):
    keys = list(first_of_l.keys())
    if len(keys) > i:
        ax.imshow(first_of_l[keys[i]], cmap='gray')  # Display each image in grayscale
        ax.set_title(keys[i],fontsize=10)
    # ax.axis('off')  Hide the axis for a cleaner look

# Display the grid of images
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()


