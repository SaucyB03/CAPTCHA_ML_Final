# load needed libraries.
import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import cv2

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

img_num = 0

#Converts images to greyscale
for img in range(len(image_files)):
    gray_image = cv2.cvtColor(images[img], cv2.COLOR_BGR2GRAY)
    _, binar = cv2.threshold(gray_image, IMG_THRESHHOLD / 255, 1, cv2.THRESH_BINARY)
    new_img.append(binar)


edge = np.array(new_img[0].copy())
plt.imshow(edge)
plt.show()
b_edge = edge.astype('uint8')

image = np.array(images[0].copy())
b_image = image.astype('uint8')


contours, hierarchy = cv2.findContours(b_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Canny Edges After Contouring', b_edge)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.show()

# for img in range(len(image_files)):
#     name = image_files[img][8:13]
#     new_img.append([])
#     for w in range(30, 135):
#         new_img[img].append([])
#         pixel_density = 0
#         for h in range(images.shape[1]):
#
#             if images[img][w][h][0] < IMG_THRESHHOLD / 255:
#                 pixel_density += 1 - images[img][w][h][0]
#
#
#         if pixel_density < 10:
#             print("ok")
#             for h in range(images.shape[1]):
#                 new_img[img][w-30].append([1,0,0])
#         else:
#             for h in range(images.shape[1]):
#                 new_img[img][w-30].append(images[img][w][h])
#
#
#
#         img_num += 1
#         print("Image ", img_num, "/", len(images)*5, " successfully converted to greyscale")
#     # img_num += 1
#
# plt.imshow(new_img[0])
# plt.show()

# new_img = numpy.array(new_img)
# print(new_img.shape)
#
# # For outputting some of new_image's characters
# fig, axes = plt.subplots(4, 5, figsize=(10, 8))  # Adjust figsize as needed
#
#
# # Loop through the images and corresponding axes to display them
# ind = 0
# for i, ax in enumerate(axes.flat):
#     for x in range(5):
#         # Calculate the start and end index for the current section
#         start_idx = x * 50 * (imagew[x][1] - imagew[x][0])
#         end_idx = (x + 1) * 50 * (imagew[x][1] - imagew[x][0])
#         # Reshape only the appropriate section of new_img[i]
#         reshaped_image = np.reshape(new_img[i], (imagew[x][1] - imagew[x][0], 50)).T
#         ax.imshow(reshaped_image, cmap='gray', interpolation="nearest")  # Display each image section in grayscale
#         ax.set_title(names[i])
#
# # Display the grid of images
# plt.tight_layout()  # Adjust layout to avoid overlap
# plt.show()
#
#
#
# # train_images, test_images, train_labels, test_labels = train_test_split()
#
#
# x_train, x_test, y_train, y_test = train_test_split(new_img, names, test_size = 0.25, random_state = 0)
#
# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
#
# print("Training x:", x_train.shape,"y:", y_train.shape)
# print("Testing x:", x_test.shape,"y:", y_test.shape)
#
# # scores = []
# # for i in range(1, x_train.shape[0], 52):
# #     print("Iteration: ", i, ", Begin...")
# #     #add your code here.
# #     treeI = RandomForestClassifier(max_features = i)
# #     treeI.fit(x_train, y_train)
# #     scores.append(treeI.score(x_test, y_test))
# #     print("Iteration:", i, ", Complete.")
# #
# # plt.plot(scores)
# # plt.show()
#
# tree = DecisionTreeClassifier()
# tree.fit(x_train, y_train)
# print(tree.score(x_test, y_test))
#
# forest = RandomForestClassifier()
# forest.fit(x_train, y_train)
# print(forest.score(x_test,y_test))
#
# max_features_range = range(1, 100, 5)  # Limit max_features to a reasonable number
#
# # # Each graph
# # scores = []
# # # Iterate over a valid range of max_features
# # for i in max_features_range:
# #     treeI = DecisionTreeClassifier(max_features=i)
# #     treeI.fit(x_train, y_train)
# #     scores.append(treeI.score(x_test, y_test))
# # # Label and plot
# # plt.plot(max_features_range, scores, label='Decision Tree Scores')
# # plt.xlabel('Max Features')
# # plt.ylabel('Accuracy')
# # plt.title('Decision Tree Performance')
# # plt.legend()
# # plt.show()
#
# scores = []
# # Iterate over a valid range of max_features
# for i in max_features_range:
#     treeI = RandomForestClassifier(max_features=i)
#     treeI.fit(x_train, y_train)
#     scores.append(treeI.score(x_test, y_test))
#     print("Identifier:",i,"Completed")
# # Label and plot
# plt.plot(max_features_range, scores, label='Random Forest Scores')
# plt.xlabel('Max Features')
# plt.ylabel('Accuracy')
# plt.title('Random Forest Performance')
# plt.legend()
# plt.show()