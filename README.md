# Customer Segmentation using K-Means Clustering

## Overview

This project applies **K-means clustering** to segment customers based on their **Annual Income (k$)** and **Spending Score (1-100)**. The dataset contains customer details from a mall, and the goal is to group similar customers into segments to inform marketing strategies.

## Dataset

The dataset used for this project is `Mall_Customers.csv` and contains the following columns:

| Column                     | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| **CustomerID**              | Unique identifier for each customer.                          |
| **Gender**                  | Gender of the customer (Male/Female).                         |
| **Age**                     | Age of the customer.                                          |
| **Annual Income (k$)**      | Annual income of the customer (in thousands of dollars).      |
| **Spending Score (1-100)**  | Spending score assigned by the mall, based on behavior (1-100).|

### Example Data:

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|--------------------|------------------------|
| 1          | Male   | 19  | 15                 | 39                     |
| 2          | Male   | 21  | 15                 | 81                     |
| 3          | Female | 20  | 16                 | 6                      |
| 4          | Female | 23  | 16                 | 77                     |
| 5          | Female | 31  | 17                 | 40                     |

## Steps to Run the Code

### 1. Install Dependencies

To run the code, you will need the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
### 2. Load and Explore the Data
You can load the data using pandas and perform basic exploratory data analysis using:

```python
Copy code
import pandas as pd

data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.describe())
```
### 3. Data Visualization
We use Seaborn's pairplot to explore relationships between the features:

```python
Copy code
import seaborn as sns

sns.pairplot(data)
This generates scatter plots of all numerical pairs in the dataset to help visualize any trends or clusters.
```

### 4. K-means Clustering
We use the Elbow Method to determine the optimal number of clusters for K-means:

```python
Copy code
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()
```
### 5. Apply K-means Clustering
Based on the Elbow Method, we choose the optimal k (e.g., k=5 clusters) and apply K-means clustering:

```python
Copy code
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)
```
### 6. Visualize the Clusters
We visualize the clusters and their centroids with a scatter plot:

```python
Copy code
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```
### 7. Results
This step segments customers into distinct groups, making it possible to target specific clusters for personalized marketing or further analysis.

# Conclusion
This project demonstrates how to perform customer segmentation using K-means clustering based on purchasing behavior. The insights gained from clustering can help businesses develop targeted strategies to improve customer engagement and sales.

File Structure
bash
Copy code
- Mall_Customers.csv    # The input dataset
- customer_segmentation.py  # Python script for data analysis and clustering
- README.md             # Project overview and instructions
