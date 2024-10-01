from sklearn.decomposition import PCA
import numpy as np
import coreSteps1

# Generate some example data
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features

# Create a PCA object
pca = PCA(n_components=2)  # Reduce to 2 components

# Fit PCA to the data and transform the data
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Print the transformed data shape
print("Transformed Data Shape:", X_pca.shape)




# Create a PCA object
pca_nitrate = PCA(n_components=5)  # Reduce to 2 components

# Fit PCA to the data and transform the data
X_pca_water = pca_nitrate.fit_transform(coreSteps1.col2)

# Print explained variance ratio
print("Explained Variance Ratio:", pca_nitrate.explained_variance_ratio_)

# Print the transformed data shape
print("Transformed Data Shape:", X_pca_water.shape)





################################################

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features (recommended for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate the PCA object
pca = PCA(n_components=2)  # Reduce to 2 components for visualization

# Fit PCA to the scaled data
pca.fit(X_scaled)

# Transform the data into the new feature space
X_pca = pca.transform(X_scaled)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Plot the PCA components
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca)
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()




################################################



# Load the Iris dataset
# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 

# Standardize the features (recommended for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate the PCA object
pca = PCA(n_components=5)  # Reduce to 2 components for visualization

# Fit PCA to the scaled data
pca.fit(X_scaled)

# Transform the data into the new feature space
X_pca = pca.transform(X_scaled)

X_pca


# Create a DataFrame for visualization
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2','PC3', 'PC4','PC5'])
df_pca['target'] = y

# Plot the PCA components
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca)
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()









