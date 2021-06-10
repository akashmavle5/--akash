#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
dataset = pd.read_csv('Akash001-Nuskin.csv')
#Adding the columns we need to work with
X = dataset.iloc[:,[3,4]].values 

#Finding optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting k-means to the dataset akash.mavle@mlogica.com, Mar 2021
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#Plotting all values of customer data(income and score) on a scatterplot
plt.scatter(X[:,0],X[:,1])
plt.title('Customers data')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 70, c = 'red', label = 'Target')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 70, c = 'blue', label = 'Motivate')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 70, c = 'green', label = 'Discount')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 70, c = 'cyan', label = 'Fans')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 70, c = 'magenta', label = 'Casual')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'orange', label = 'Centroids')
plt.title('Sales Volume Cluster for NuSkin')
plt.xlabel('Sales  Volume(k$)-Created by Akash')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()