#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sarvandani
"""
#data can be found at https://www.kaggle.com/datasets/maralka/credit-card-customer
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
# import missingno

#############################################
#reading data
Data = pd.read_csv("Customer_Data-kaggle.csv")

##############################################
#checking missing values
null_values = Data.isnull().sum()
##################################
#checking duplicated values
duplicate_values = Data[Data.duplicated()]
################################
##rectifying missing values
df = pd.DataFrame(Data)
## customer ID must be dropped
df.drop('ID', axis=1,inplace=True)
fill_cols = [col for col in df.columns]
itr_imputer = IterativeImputer(initial_strategy='median', min_value=0, random_state=100)
df[fill_cols] = itr_imputer.fit_transform(df[fill_cols])
plt.figure(figsize = (9,6))
sns.heatmap(df.isnull(), cmap= 'PiYG', cbar=False, yticklabels=False, xticklabels=df.columns)
###################################################
## data analysis
##correlations: high correlations are in red
correlation = df.corr(method='pearson')
fig, ax = plt.subplots()
ax.figure.set_size_inches(20, 20)
# a mask for the upper triangle
mask = np.triu(np.ones_like(correlation, dtype=np.bool))
# plots the coorelations
sns.heatmap(correlation, cmap='rainbow', mask=mask, square=True, linewidths=.5, annot=True, annot_kws={'size':14})
plt.show()
##################################
##data analysis
fig1, ax = plt.subplots(len(fill_cols),1,figsize=(10,50))
for i, col in enumerate(df):
    sns.histplot(df[col], kde=True, ax=ax[i], color='green', bins = 30)
fig1.tight_layout()
plt.show()
#########################
fig2, ax = plt.subplots(len(fill_cols),1,figsize=(10,50))
for i, col in enumerate(df):
    sns.kdeplot(df[col], ax=ax[i], color='red', multiple="stack")
fig2.tight_layout()
plt.show()
#############################
#############################################
## scaling 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
plt.figure(figsize=(20,9))
sns.heatmap(scaled_features)
plt.show()
#######################################################
## Clustering methods: kmean
##The first step is to randomly choose k centroids, where k is number of clusters
##The random initialization step causes the k-means algorithm to be  varied if you run the same algorithm twice on the same data
kmeans_set = {"init":"random", "max_iter":200,"random_state":42}
## next step we should define the number of clusters by a method here I used silhouette_coefficients.
silhouette_coefficients =[]
for k in range(2,len(fill_cols)+1):
    kmeans=KMeans(n_clusters=k,**kmeans_set).fit(scaled_features)
    score=silhouette_score(scaled_features,kmeans.labels_)
    silhouette_coefficients.append(score)
plt.style.use("fivethirtyeight")
plt.plot(range(2,len(fill_cols)+1),silhouette_coefficients,marker='o')
plt.xticks(range(2,len(fill_cols)+1))
plt.xlabel("Number of Clusters")
plt.ylabel("silhouette coefficients")
plt.show()
############
##model
# 3 cluster was chosen based on silhouette analysis
kmeans = KMeans(n_clusters=3,**kmeans_set).fit(scaled_features)
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df.columns])
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df.columns])
##inja: ehtemalan bayad jabeja beshe mahale labels = kmeans.labels_ va  df_cluster_with_kmean adding cluster to data
labels = kmeans.labels_
df_cluster_with_kmean = pd.concat([df, pd.DataFrame({'cluster': labels})], axis = 1)
ax = sns.countplot(data=df_cluster_with_kmean, x='cluster', palette='Accent_r', saturation=1, linewidth = 1)
for cont in ax.containers:
    ax.bar_label(cont)
###########################################
######################@
##visualizing clustering results
kmeans = KMeans(n_clusters=3,init= "random", random_state = 1).fit(df)
centroids = kmeans.cluster_centers_
plt.figure(figsize=(15,8))
df_kmean = df.copy()
df_kmean['cluster'] = kmeans.labels_
sns.relplot(data = df_kmean ,x='CASH_ADVANCE' , y  ='PURCHASES', hue='cluster', palette='Accent' ,kind='scatter', height=8.27, aspect = 11.7/8.27)
plt.scatter(centroids[:, 0], centroids[:,1], c='red', s=50)
plt.xlabel("CASH_ADVANCE",fontsize=15)
plt.ylabel("PURCHASES",fontsize=15)
###########################################
##ALL
best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE","CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS","cluster"]
sns.pairplot( df_cluster_with_kmean[ best_cols ], hue="cluster",palette='Accent')  
##################################################################################    
##clustering method, model 2: hierarchy: 
data = df
X =  data.iloc[:, [3, 6]].values 
plt.figure(figsize=(20,15))
dendrogramm = hierarchy.dendrogram(hierarchy.linkage(X, method = 'ward'),leaf_font_size=10)
plt.title('Dendrogram of Customers')
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.grid(False)
plt.xticks(rotation=90)
plt.show()  
####################################
##Determining number of clusters
sil_Hierarch = []
for k in range(2,len(fill_cols)+1):
    Hierarch = AgglomerativeClustering(n_clusters = k,linkage='ward').fit_predict(scaled_features)
    score = silhouette_score(scaled_features, Hierarch,metric='euclidean')
    sil_Hierarch.append(score)
plt.style.use("fivethirtyeight")
plt.plot(range(2,len(fill_cols)+1), sil_Hierarch)
plt.xticks(range(2,len(fill_cols)+1))
plt.ylabel("silhouette coefficients")
plt.show()  
##2 was chosen as the number of clusters
############################################
## hierearchical_model: bottom-up approach##
##this algorithm considers each dataset as a cluster then start combining the closest clusters together.
################
Hierarch = AgglomerativeClustering(n_clusters = 2).fit(df)
df_Hierarch = df.copy()
df_Hierarch['cluster'] = Hierarch.labels_
sns.relplot(data = df_Hierarch ,x='CASH_ADVANCE' , y  ='PURCHASES', hue='cluster', palette='viridis' ,kind='scatter', height=8.27, aspect = 11.7/8.27)
plt.title('Clusters')
plt.xlabel('CASH_ADVANCE')
plt.ylabel('PURCHASES')
#################
##ALL
labels = Hierarch.labels_
df_with_Hierarch = pd.concat([df, pd.DataFrame({'cluster': labels})], axis = 1)
best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE","CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS","cluster"]
sns.pairplot( df_with_Hierarch[best_cols ], hue="cluster",palette='viridis') 

#######################################
#### model3: In DBSCAN, clusters are formed from dense regions and separated by regions of no or low densities.
##detemine epsilon and minPts. 
##The minPts parameter is easy to set. The minPts should be 4 for two-dimensional dataset.
# n_neighbors = 5 as kneighbors function returns distance of point to itself
nbrs = NearestNeighbors(n_neighbors = 5).fit(df)
# Find the k-neighbors of a point to get eps.
neigh_dist, neigh_ind = nbrs.kneighbors(df)
# sort the neighbor distances (lengths to points) in ascending order
# axis = 0 represents sort along first axis i.e. sort along row
sort_neigh_dist = np.sort(neigh_dist, axis = 0)
k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()
#########################
##4500 was based on the changing point in the curve
DBSCANmodel = DBSCAN(eps = 4500, min_samples = 4).fit(df)
set(DBSCANmodel.labels_) 
# -1 value represents noisy points could not assigned to any cluster
p = sns.scatterplot(data = df, x = "CASH_ADVANCE", y = "PURCHASES", hue = DBSCANmodel.labels_, legend = "full", palette = "icefire")
sns.move_legend(p, "upper right", bbox_to_anchor = (1.17, 1.), title = 'Clusters')
plt.show()
#####################################
##all
labels = DBSCANmodel.labels_
df_with_DBSCAN = pd.concat([df, pd.DataFrame({'cluster': labels})], axis = 1)
df_DBSCANmodel = df.copy()
df_DBSCANmodel['cluster'] = DBSCANmodel.labels_
sns.pairplot( df_with_DBSCAN[ best_cols ], hue="cluster",palette='icefire')  
