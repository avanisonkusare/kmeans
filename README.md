Customer Segmentation using K-Means Clustering
Overview
This project applies K-means clustering to segment customers based on their Annual Income (k$) and Spending Score (1-100) from a dataset of mall customers. The objective is to identify distinct customer segments, which can be useful for targeted marketing strategies and business insights.

Dataset
The dataset used for this project is the Mall_Customers.csv, which contains the following columns:

CustomerID: A unique identifier for each customer.
Gender: The gender of the customer (Male/Female).
Age: The age of the customer.
Annual Income (k$): The annual income of the customer (in thousands of dollars).
Spending Score (1-100): A spending score assigned by the mall based on customer behavior (1 to 100).
Example Data
CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
1	Male	19	15	39
2	Male	21	15	81
3	Female	20	16	6
4	Female	23	16	77
5	Female	31	17	40
Steps to Run
1. Install Dependencies
You need to install the following Python libraries to run the code:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
2. Load and Explore the Data
The data is loaded into a pandas DataFrame, and basic exploratory data analysis is performed:

data.info() gives an overview of the data types and the number of non-null entries.
data.describe() provides summary statistics (mean, median, min, max, etc.) for the numerical columns.
3. Data Visualization
The pairplot from seaborn is used to visualize relationships between variables:

python
Copy code
sns.pairplot(data)
This step helps in visually identifying any obvious patterns or clusters in the data.

4. K-means Clustering
Using the Elbow Method, we determine the optimal number of clusters (k) for K-means:

python
Copy code
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
After determining the best k (in this case, 5 clusters), we apply the K-means algorithm:

python
Copy code
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)
5. Cluster Visualization
The final clusters are visualized on a scatter plot, where each point represents a customer, and the colors represent the different clusters. The centroids of each cluster are also plotted in red:

python
Copy code
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
6. Results
By applying K-means clustering, the customers are segmented into distinct groups based on their annual income and spending behavior. These insights can be used for targeted marketing and to understand customer behavior.

Conclusion
This project demonstrates how to use K-means clustering for customer segmentation based on purchasing patterns. By identifying groups of customers with similar behavior, businesses can design personalized marketing campaigns and optimize product offerings.
