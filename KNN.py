
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Step 1: Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Total flowers:", len(x))
print("Total measurements:", x.shape[1])
print("Measurement names:", iris.feature_names)
print("Flower species:", iris.target_names)

print("\\\\nFirst flower measurement:", x[0])
print("First flower species:", iris.target_names[y[0]])


# Step 2: Split the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\\\\nSTEP 2: Dataset Split")
print("Training flowers:", len(x_train))
print("Testing flowers:", len(x_test))
print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))


# Step 3: Calculate Euclidean distance
def flower_distance(a, b):
    difference = a - b
    squared_difference = difference ** 2
    total = np.sum(squared_difference)
    distance = np.sqrt(total)
    return distance


# Test the distance function using two flowers
flower1 = x_train[0]
flower2 = x_train[1]

distance = flower_distance(flower1, flower2)

print("\\\\nSTEP 3: Distance Calculation")
print("Flower 1:", flower1)
print("Flower 2:", flower2)
print("Distance:", distance)


# Step 4: Find the nearest neighbors
def find_neighbors(test_flower, x_train, k):
    distances = []

    for i in range(len(x_train)):
        distance = flower_distance(test_flower, x_train[i])
        distances.append((distance, i))

    # Sort after checking all training flowers
    distances.sort()

    # Return the k closest flowers
    return distances[:k]


# Test Step 4
neighbors = find_neighbors(x_test[0], x_train, 3)

print("\\\\nSTEP 4: Nearest Neighbors")
for distance, index in neighbors:
    print("Distance:", round(distance, 3))
    print("Species:", iris.target_names[y_train[index]])


# Step 5: Predict the flower species using majority vote
def predict_species(test_flower, x_train, y_train, k):
    # Find the nearest k flowers
    neighbors = find_neighbors(test_flower, x_train, k)

    # Store the species of the neighbors
    neighbor_labels = []

    for distance, index in neighbors:
        neighbor_labels.append(y_train[index])

    # Count votes for each species
    counts = np.bincount(neighbor_labels)

    # Select the species with the most votes
    prediction = np.argmax(counts)

    return prediction


# Test Step 5 with one flower
prediction = predict_species(x_test[0], x_train, y_train, k=3)

print("\\\\nSTEP 5: Flower Prediction")
print("Predicted species:", iris.target_names[prediction])
print("Actual species:", iris.target_names[y_test[0]])


# Step 6: Predict all testing flowers and print the classification report
y_pred = []

for test_flower in x_test:
    prediction = predict_species(test_flower, x_train, y_train, k=3)
    y_pred.append(prediction)

y_pred = np.array(y_pred)

print("\\\\nSTEP 6: Classification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# Step 7: Create a 2D scatter plot
# X-axis: Sepal length
# Y-axis: Petal length
# Use y_test for actual species colors, or y_pred for predicted species colors.
# Set this to True to color dots by actual species.
plot_actual_species = True

labels_to_plot = y_test if plot_actual_species else y_pred

plt.figure(figsize=(8, 6))

for species in range(3):
    indexes = np.where(labels_to_plot == species)[0]

    plt.scatter(
        x_test[indexes, 0],
        x_test[indexes, 2],
        label=iris.target_names[species]
    )

plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.title("KNN Classification (2D view)")
plt.legend()
plt.show()


# Step 8: Create a 3D scatter plot
# X-axis: Sepal length
# Y-axis: Sepal width
# Z-axis: Petal length
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for species in range(3):
    indexes = np.where(labels_to_plot == species)[0]

    ax.scatter(
        x_test[indexes, 0],
        x_test[indexes, 1],
        x_test[indexes, 2],
        label=iris.target_names[species]
    )

ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Sepal width (cm)")
ax.set_zlabel("Petal length (cm)")
ax.set_title("KNN Classification (3D view)")
ax.legend()
plt.show()


# Step 9: Show any flowers that were misclassified
print("\\\\nSTEP 9: Misclassified Flowers")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(
            f"Test flower {i}: "
            f"Actual = {iris.target_names[y_test[i]]}, "
            f"Predicted = {iris.target_names[y_pred[i]]}"
        )


# Step 10: Questions and Answers

# Question 1: What are the disadvantages of KNN?
# Answer:
# KNN can be slow for large datasets because it calculates
# distances to many training samples for every prediction.
# It is sensitive to irrelevant features and the choice of K.
# It can also perform poorly when classes are imbalanced.

# Question 2: How do we choose the value of K?
# Answer:
# We can test different K values and compare their accuracy
# using validation data.
# A small K can be sensitive to noise.
# A large K may include too many distant neighbors.
# A common starting point is the square root of the number
# of training samples, but validation is more reliable.

# Question 3: What other distance measures can be used?
# Answer:
# Manhattan distance: adds the absolute differences between
# features. It can be useful for grid-like or city-block data.

# Minkowski distance: a general distance measure that includes
# Euclidean and Manhattan distance as special cases.

# Hamming distance: counts differing positions. It is useful
# for binary features or categorical data encoded as strings.

