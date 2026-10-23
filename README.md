# K-Nearest Neighbors (KNN) - Iris Flower Classification

## Project Description
This project implements a **K-Nearest Neighbors (KNN)** machine learning algorithm from scratch using Python and the Iris dataset. The goal is to classify Iris flowers into three species: Setosa, Versicolor, and Virginica, based on their measurements.

Instead of using a built-in KNN classifier, this project demonstrates the core logic of KNN, including calculating Euclidean distances, finding the nearest neighbors, and predicting a flower species through majority voting.

## Features
- Loads and explores the Iris dataset using Scikit-learn.
- Splits the dataset into training and testing sets.
- Implements Euclidean distance calculation using NumPy.
- Finds the K nearest neighbors for a given flower.
- Predicts flower species using majority voting.
- Evaluates predictions using a classification report.
- Visualizes classification results using 2D and 3D scatter plots.
- Identifies flowers that were misclassified.

## Technologies Used
- Python
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
The Iris dataset contains 150 flower samples from three species:

- Setosa
- Versicolor
- Virginica

Each flower has four measurements:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

## How KNN Works
1. Load the Iris dataset and divide it into training and testing data.
2. Calculate the Euclidean distance between a test flower and each training flower.
3. Select the K closest flowers (K = 3 in this project).
4. Count the species labels of the selected neighbors.
5. Predict the species with the highest number of votes.
6. Evaluate the model by comparing predicted labels with actual labels.

## Installation and Usage

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd YOUR_PROJECT_FOLDER
```

Install the required libraries:

```bash
pip install numpy scikit-learn matplotlib
```

Run the Python script:

```bash
python knn_iris_recreated.py
```

## Visualization
The project generates:
- A 2D scatter plot using sepal length and petal length.
- A 3D scatter plot using sepal length, sepal width, and petal length.

The plots can display actual flower species or KNN-predicted species.

## Learning Objectives
This project helped me understand:
- How the KNN classification algorithm works.
- How distance-based classification is implemented.
- The importance of training and testing datasets.
- How to evaluate a classification model.
- How to visualize machine learning predictions.

## Author
Muhammad Shoban
## Results and Visualizations

### 2D KNN Classification

![2D KNN Classification](images/Figure_1.png)

### 3D KNN Classification

![3D KNN Classification](images/Figure_2.png)
