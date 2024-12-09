import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml

# download the dataset 
mnist = fetch_openml('mnist_784', version=1)


X = mnist.data
y = mnist.target.astype(int)

# Shapes that are used for pattern
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# Display samples images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.show()

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Class PCA
pca = PCA()

# Use PCA on the standardized data
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.grid()
plt.show()

# Choose number of components to retain 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced number of features: {X_pca.shape[1]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression on PCA-reduced data: {accuracy:.2f}")

# Display PCA components
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(pca.components_[i].reshape(28, 28), cmap='viridis')
    ax.set_title(f"PCA Component {i+1}")
    ax.axis('off')
plt.show()
