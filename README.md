# Machine_learning_clustering_Credit_Card_Customer
Cluster analysis is a machine learning approach that involves grouping unlabelled data based on shared characteristics. Essentially, it involves dividing a dataset into clusters of data points that are similar to each other, while being dissimilar to those in other clusters.
To accomplish this, the algorithm seeks out patterns within the unlabelled dataset, such as shape, size, color, or behavior, and then groups data points based on the presence or absence of those patterns. Unlike supervised learning methods, cluster analysis is an unsupervised technique that doesn't require labeled data. Once the algorithm has completed the clustering process, each group is assigned a unique cluster ID, which can be used to simplify the processing of large and complex datasets by the machine learning system.

I have used threes approaches of clustering method on the data of credit card. The steps of the works are as follows:

I check and rectify the missing values in the dataset:

<img src="missing.png" >

The correlations between the parameters of the dataset is analyzed:

<img src="correlation.png" >

The histograms of two parameters of the dataset have been visualized.

<img src="balancee.png" >

<img src="credit_limit.png" >

The dataset has been scaled:

<img src="scaled.png" >

Now, we are able to apply the clustering methods. 

1. In data mining, the K-means algorithm is utilized to handle learning data. Initially, a set of random centroids is selected and assigned to each cluster. The algorithm then iteratively performs calculations to optimize the positions of the centroids until they become stable or a defined number of iterations is reached. At this point, the creation and optimization of clusters stop, indicating a successful clustering process.
Important step is to determine the number of clusters. I used Silhouette Coefficient.

<img src="kmean_cluster.png" >

3 clusters were chosen based on silhouette analysis. 

Clustering results by K-mean between two variables:

<img src="kmean_pair_cluster.png" width="600" height="400">


2. The Agglomerative Hierarchical Clustering Technique involves treating each data point as a separate cluster at the beginning. Through a series of iterations, similar clusters are combined with one another until a single cluster or K clusters are formed.

Important step is to determine the number of clusters. I used Silhouette Coefficient.

2 clusters were chosen based on silhouette analysis. 

<img src="hierarchy_cluster_number.png" >

Clustering results by Agglomerative Hierarchical Clustering Technique between two variables:

<img src="hierarchy_pair_cluster.png"  width="600" height="400" >














