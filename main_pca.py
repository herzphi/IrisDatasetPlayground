import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

plt.set_cmap("Paired")

iris = datasets.load_iris()

data = iris.data
target = iris.target


mean = np.mean(data)
std = np.std(data)
Z = (data - mean) / std

"""
    The covariance matrix C describes the variance
    and covariance between pairs of variables.
"""
n = len(Z)
C = np.dot(np.transpose(Z), Z) / (n - 1)
"""
    The eigenvalues give the magnitude of the variance explained
    by each principal component, and the eigenvectors give the
    direction of these components.
"""
# Calculate the eigenvalues, eigenvectors
# and sort them by ascending eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(C)
sorted_indices = np.argsort(eigenvalues)[::-1]

# Sort the eigenvalues and eigenvectors
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

X_pca_0 = np.dot(Z, sorted_eigenvectors)
# Compute the cumulative explained variance
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance)
# Select the number of components that explain the desired amount of variance
desired_variance = 0.95
# Select the number of components that explain the desired amount of variance
num_components = np.argmax(cumulative_explained_variance >= desired_variance) + 1
principal_components = sorted_eigenvectors[:, :num_components]
X_pca_red = np.dot(Z, principal_components)

# Plotting the categories in color
_, axs = plt.subplots(3, figsize=(8, 8))


def plot_scatter(ax, data, x_id, y_id):
    scatter = ax.scatter(data[:, x_id], data[:, y_id], c=iris.target)
    ax.set(xlabel=iris.feature_names[x_id], ylabel=iris.feature_names[y_id])
    _ = ax.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower right",
        title="Classes",
    )


for ax, data in zip(axs.flat, [X_pca_red, iris.data, X_pca_0]):
    plot_scatter(ax, data, x_id=0, y_id=1)
plt.show()
