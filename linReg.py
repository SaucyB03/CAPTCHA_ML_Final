# # load needed libraries.
# import os
# import numpy
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
# import glob
#
# # Functions
# def calcVectorizedCost(X, Y, theta):
#     inner = np.dot(((np.dot(X, theta)) - Y).T, (np.dot(X, theta)) - Y)
#     return inner / (2 * len(X))
#
# def gradientDescent(X, Y, theta, eta, iters):
#     cost = np.zeros(iters)
#     for i in range(iters):
#         gradients = 2 * (np.dot(X.T, ((np.dot(X, theta))) - Y)/ (len(X)))
#         theta = theta - eta * gradients
#         cost[i] = calcVectorizedCost(X, Y, theta)
#     return theta, cost
#
# def GeneratePolynomialFeatures(X, polydegree):
#     poly = PolynomialFeatures(polydegree)# create sklearn PolynomialFeatures Object with degree=polydegree
#     polynomial_x = poly.fit_transform(X)# This is the new generated features.
#     return polynomial_x
#
# def plot_SimpleNonlinearRegression_line(theta, X, poly):
#     # find min and max values
#     min_x = np.min(X)
#     max_x = np.max(X)
#     # get range of data to transform and make predictions on
#     min_max_range = np.linspace(min_x, max_x, 100)
#     min_max_range = min_max_range.reshape(-1, 1)
#     # transform data
#     polynomial_min_max_range = poly.fit_transform(min_max_range)
#     y_vals = np.dot(polynomial_min_max_range, theta)
#     plt.plot(min_max_range, y_vals)
# #############################
#
#
# IMG_THRESHHOLD = 170
#
# # Loads images, converts to numpy array, and displays
# image_files = glob.glob(os.path.join("samples", '*.png'))
# images = numpy.array([plt.imread(img) for img in image_files])
# print(images.shape)
# plt.imshow(images[0])
# plt.show()
# # Processing the images
#
# #grayscale versions of images
# new_img = []
# #list of names of each image
# names = []
# #image width and height
# imagew = [
#     [30,55],
#     [50, 75],
#     [70, 95],
#     [90, 115],
#     [110, 135]
#     ]
#
# img_num = 0
#
# #Converts images to greyscale
# for img in range(len(image_files)):
#     temp = image_files[img][8:13]
#     for a in range(5):
#         new_img.append([])
#         names.append(temp[a])
#         for w in range(imagew[a][0], imagew[a][1]):
#             for h in range(images.shape[1]):
#                 if images[img][h][w][0] < IMG_THRESHHOLD/255:
#                     new_img[img_num].append(images[img][h][w][0])
#                 else:
#                     new_img[img_num].append(1.0)
#
#         img_num += 1
#         print("Image ", img_num, "/", len(images)*5, " successfully converted to greyscale")
#     # img_num += 1
#
# new_img = numpy.array(new_img)
# print(new_img.shape)
#
# X = new_img / 255.0
# Y = names
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#
# # x_train = np.array(x_train)
# # x_test = np.array(x_test)
# # y_train = np.array(y_train)
# # y_test = np.array(y_test)
#
# label_encoder = LabelEncoder()
# # Encode the target labels (characters) into numeric values
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)
#
# print("Training x:", x_train.shape,"y:", y_train_encoded.shape)
# print("Testing x:", x_test.shape,"y:", y_test_encoded.shape)
#
# # Perform polynomial regression
# polydegree = 5
# poly = PolynomialFeatures(degree=polydegree)
# x_train_poly = poly.fit_transform(x_train)
# x_test_poly = poly.transform(x_test)
#
# model = LinearRegression(fit_intercept=False)
# model.fit(x_train_poly, y_train_encoded)
#
# # Evaluate the model
# r2_score = model.score(x_test_poly, y_test)
# print(f"Polynomial Regression R² score (degree={polydegree}): {r2_score:.4f}")
#
# # Plot regression results for a single feature
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values')
# plt.scatter(range(len(y_test)), model.predict(x_test_poly), color='red', alpha=0.6, label='Predicted Values')
# plt.title("Polynomial Regression Results")
# plt.xlabel("Sample Index")
# plt.ylabel("Label Value")
# plt.legend()
# plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# # Cost function
# def calcVectorizedCost(X, Y, theta):
#     m = len(Y)
#     predictions = np.dot(X, theta)
#     cost = np.sum((predictions - Y) ** 2) / (2 * m)
#     return cost
#
# # Gradient descent
# def gradientDescent(X, Y, theta, eta, iters):
#     m = len(Y)
#     cost_history = np.zeros(iters)
#
#     for i in range(iters):
#         gradients = (np.dot(X.T, (np.dot(X, theta) - Y))) / m
#         theta -= eta * gradients
#         cost_history[i] = calcVectorizedCost(X, Y, theta)
#
#     return theta, cost_history
#
# # Load images and convert them to grayscale
# IMG_THRESHHOLD = 170
# image_files = [os.path.join("samples", img) for img in os.listdir("samples") if img.endswith(".png")]
# images = np.array([plt.imread(img) for img in image_files])
# print(f"Loaded {len(images)} images of shape {images[0].shape}")
#
# # Preprocess images into grayscale features
# new_img = []
# names = []
#
# for img_num, img in enumerate(images):
#     grayscale = img.mean(axis=-1)  # Convert to grayscale
#     flattened = grayscale.flatten()
#     new_img.append(flattened)
#     names.append(f"Image_{img_num}")
#
# new_img = np.array(new_img)
# print("Processed image array shape:", new_img.shape)
#
# # Split data into training and test sets
# X = new_img / 255.0  # Normalize pixel values
# Y = np.arange(len(names))  # Assign numeric labels to each image (dummy labels)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#
# # Polynomial feature generation
# polydegree = 2
# poly = PolynomialFeatures(degree=polydegree)
# x_train_poly = poly.fit_transform(x_train)
# x_test_poly = poly.transform(x_test)
#
# # Initialize gradient descent parameters
# theta = np.zeros((x_train_poly.shape[1], 1))  # Initialize weights
# y_train = y_train.reshape(-1, 1)  # Reshape for matrix operations
# eta = 0.01  # Learning rate
# epochs = 100  # Number of iterations
#
# # Train the model using gradient descent
# theta, cost_history = gradientDescent(x_train_poly, y_train, theta, eta, epochs)
#
# # Evaluate the model
# train_cost = calcVectorizedCost(x_train_poly, y_train, theta)
# test_cost = calcVectorizedCost(x_test_poly, y_test.reshape(-1, 1), theta)
# print(f"Training cost: {train_cost:.4f}")
# print(f"Test cost: {test_cost:.4f}")
#
# # Plot cost history
# plt.figure(figsize=(10, 6))
# plt.plot(range(epochs), cost_history, color='blue', label='Cost History')
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.title("Gradient Descent Cost Over Iterations")
# plt.legend()
# plt.show()
#
# # Plot regression results for a single feature
# predictions = np.dot(x_test_poly, theta).flatten()
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values')
# plt.scatter(range(len(y_test)), predictions, color='red', alpha=0.6, label='Predicted Values')
# plt.title("Polynomial Regression Results")
# plt.xlabel("Sample Index")
# plt.ylabel("Label Value")
# plt.legend()
# plt.show()

# # Import required libraries
# import os
# import numpy as np
# import glob
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
#
# # Constants
# IMG_THRESHOLD = 170
#
# # Load images and convert them to numpy arrays
# image_files = glob.glob(os.path.join("samples", '*.png'))
# images = np.array([plt.imread(img) for img in image_files])
# print(f"Loaded {len(images)} images of shape {images[0].shape}")
#
# # Process images into grayscale feature arrays
# new_img = []
# names = []
# image_widths = [
#     [30, 55],
#     [50, 75],
#     [70, 95],
#     [90, 115],
#     [110, 135]
# ]
# img_num = 0
#
# for img in range(len(image_files)):
#     temp = image_files[img][8:13]
#     for a in range(5):
#         new_img.append([])
#         names.append(temp[a])
#         for w in range(image_widths[a][0], image_widths[a][1]):
#             for h in range(images.shape[1]):
#                 if images[img][h][w][0] < IMG_THRESHOLD / 255:
#                     new_img[img_num].append(images[img][h][w][0])
#                 else:
#                     new_img[img_num].append(1.0)
#         img_num += 1
#
# new_img = np.array(new_img)
# print(f"Processed image array shape: {new_img.shape}")
#
# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(new_img, names, test_size=0.25, random_state=0)
# x_train, x_test = np.array(x_train), np.array(x_test)
# y_train, y_test = np.array(y_train), np.array(y_test)
#
# # Encode labels
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)
#
# print(f"Training x: {x_train.shape}, y: {y_train_encoded.shape}")
# print(f"Testing x: {x_test.shape}, y: {y_test_encoded.shape}")
#
# # PCA for dimensionality reduction
# pca = PCA(n_components=100)  # Reduce to 100 components
# x_train_reduced = pca.fit_transform(x_train)
# x_test_reduced = pca.transform(x_test)
# print(f"x_train_reduced shape: {x_train_reduced.shape}, x_test_reduced shape: {x_test_reduced.shape}")
#
# # Polynomial feature generation (with batching to avoid memory overload)
# def generate_polynomial_features_in_batches(data, degree, batch_size=100):
#     poly = PolynomialFeatures(degree=degree)
#     processed_batches = []
#     for i in range(0, data.shape[0], batch_size):
#         batch = data[i:i+batch_size]
#         batch_poly = poly.fit_transform(batch)
#         processed_batches.append(batch_poly)
#     return np.vstack(processed_batches)
#
# polydegree = 2  # Set the desired polynomial degree
# x_train_poly = generate_polynomial_features_in_batches(x_train_reduced, degree=polydegree)
# x_test_poly = generate_polynomial_features_in_batches(x_test_reduced, degree=polydegree)
# print(f"x_train_poly shape: {x_train_poly.shape}, x_test_poly shape: {x_test_poly.shape}")
#
# # Train linear regression model
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression(fit_intercept=False)
# model.fit(x_train_poly, y_train_encoded)
# print(f"Model R^2 score on test data: {model.score(x_test_poly, y_test_encoded)}")

# Import required libraries
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Constants
IMG_THRESHOLD = 170

# Load images and convert them to numpy arrays
image_files = glob.glob(os.path.join("samples", '*.png'))
images = np.array([plt.imread(img) for img in image_files])
print(f"Loaded {len(images)} images of shape {images[0].shape}")

# Process images into grayscale feature arrays
new_img = []
names = []
image_widths = [
    [30, 55],
    [50, 75],
    [70, 95],
    [90, 115],
    [110, 135]
]
img_num = 0

for img in range(len(image_files)):
    temp = image_files[img][8:13]
    for a in range(5):
        new_img.append([])
        names.append(temp[a])
        for w in range(image_widths[a][0], image_widths[a][1]):
            for h in range(images.shape[1]):
                if images[img][h][w][0] < IMG_THRESHOLD / 255:
                    new_img[img_num].append(images[img][h][w][0])
                else:
                    new_img[img_num].append(1.0)
        img_num += 1

new_img = np.array(new_img)
print(f"Processed image array shape: {new_img.shape}")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(new_img, names, test_size=0.25, random_state=0)
x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"Training x: {x_train.shape}, y: {y_train_encoded.shape}")
print(f"Testing x: {x_test.shape}, y: {y_test_encoded.shape}")

# PCA for dimensionality reduction
pca = PCA(n_components=100)  # Reduce to 100 components
x_train_reduced = pca.fit_transform(x_train)
x_test_reduced = pca.transform(x_test)
print(f"x_train_reduced shape: {x_train_reduced.shape}, x_test_reduced shape: {x_test_reduced.shape}")

# Polynomial feature generation (with batching to avoid memory overload)
def generate_polynomial_features_in_batches(data, degree, batch_size=100):
    poly = PolynomialFeatures(degree=degree)
    processed_batches = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        batch_poly = poly.fit_transform(batch)
        processed_batches.append(batch_poly)
    return np.vstack(processed_batches)

# Evaluate model across different polynomial degrees
poly_degrees = range(1, 3)  # Degrees to evaluate
r2_scores = []

for degree in poly_degrees:
    print(f"Processing polynomial degree: {degree}")
    x_train_poly = generate_polynomial_features_in_batches(x_train_reduced, degree=degree)
    x_test_poly = generate_polynomial_features_in_batches(x_test_reduced, degree=degree)

    model = LinearRegression(fit_intercept=False)
    model.fit(x_train_poly, y_train_encoded)
    r2_score = model.score(x_test_poly, y_test_encoded)
    r2_scores.append(r2_score)
    print(f"R² score for degree {degree}: {r2_score}")

# Plot R² scores for different polynomial degrees
plt.figure(figsize=(8, 6))
plt.plot(poly_degrees, r2_scores, marker='o', linestyle='-', color='b', label='R² Score')
plt.title('Model Accuracy vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
