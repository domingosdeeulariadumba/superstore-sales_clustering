# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 06:12:10 2023

@author: domingosdeeularia
"""
# %%



# %%
""" IMPORTING LIBRARIES """
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors as Nngb
    
import warnings
warnings.filterwarnings('ignore')
# %%



# %%
"""" EXPLORATORY DATA ANALYSIS """



    '''
    We first import the dataset, and explore its information. That is the
    attributes, datatype, amount of missing values, statistical summary, the
    first and last five rows.
    '''
df = pd.read_excel("superstore_sales.xls")

df = df[df['Category'] == 'Furniture'].iloc[:,:21]

df.info()

df.head()

df.tail()

df.describe(include = 'all')

    '''
    As the objective of the firm is to understand the sales and profit 
    representation by region for Furniture products category, below it is 
    displayed a pair plot concerning this attributes, and then a bar plot.
    '''
df_region = df[['Profit', 'Sales', 'Region']]

    ''' Pair Plot'''
sb.pairplot(df_region, hue ='Region')
plt.show()

    ''' Grouped Bar Plot'''
ax = df_region.groupby(['Region']).mean().plot(kind='bar',
                                               color = ['b',
                                                        'm'])
plt.xlabel('')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 0)
plt.title('Furniture Sales and Profit by Region', 
          fontweight='bold')
plt.show()
#%%



# %%
"""" CLUSTER ANALYSIS """



    '''
    The central region is the one with lowest profits (on average there is about
    5.97 million dollars of loss). We'll focus our efforts on this region. The
    goal is to define the ideal number of clusters so the firm can implement
    customized strategies to revert the currrent situation. We start by 
    visualizing the scatter plot for profits and unit sales in this region
    '''
df_central = df_region[df_region['Region'] == 'Central']

sb.scatterplot(data = df_central, x = 'Sales', y = 'Profit')
plt.title('Profit Vs. Sales in Central Region',
          fontweight = 'bold')

    '''
    Next, we look for the ONC (Optimal Number of Clusters) in a range of two to
    ten. These clusters and their respective silhouette scores are stored in a
    dataframe. We Display this dataframe and then it is presented a line plot
    which allows to easily gridsearch the ONC (the group with the higher
    silhouette score, which represents the similarity degree of a data point
    to its own cluster compared to others)
    '''

silh_vls = []

for i in range (2,11):
    MyKmeans_silh=KMeans(n_clusters=i).fit(df_central[['Sales','Profit']])
    silh_avg=silhouette_score(df_central[['Sales','Profit']],
                              MyKmeans_silh.labels_)
    silh_vls.append(silh_avg)

df_scores = pd.DataFrame({'n_clusters': [i for i in range(2,11)],
             'silhouette_scores': silh_vls}) 
display(df_scores)

    '''
    Filtering the Optimal Number of Clusters
    '''

N_clusters = df_scores[df_scores['silhouette_scores'
                                 ] == df_scores[
                                     'silhouette_scores'
                                     ].max()]['n_clusters'].squeeze()
    
    '''
    Silhoutte Plot for KMeans Clustering equal to N_clusters
    '''
visual_silh = SilhouetteVisualizer(KMeans(n_clusters = N_clusters,
                                          random_state = 97))
visual_silh.fit(df_central[['Sales','Profit']])
visual_silh.show()


    '''
    Silhoutte Score Vs. Number of Clusters Plot
    '''

plt.plot(df_scores['n_clusters'],
         df_scores ['silhouette_scores'], 'o-', color = 'm')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters Grid Search',
          fontweight = 'bold')
plt.show()
                                     
    '''
    Customizing each cluster ID
    '''
opt_numb_clst = KMeans(n_clusters = N_clusters)
opt_numb_clst.fit(df_central[['Sales','Profit']])
opt_numb_clst.labels_
df_central['Cluster ID'] = opt_numb_clst.labels_
df_central['Cluster ID'][df_central['Cluster ID'] == 0] = 'Cluster 1'
df_central['Cluster ID'][df_central['Cluster ID'
                                    ] != 'Cluster 1'] = 'Cluster 2'

    '''
    Inserting the clusters centers
    '''
centers = pd.DataFrame(opt_numb_clst.cluster_centers_)
centers.columns = ['x','y']
  
    '''
    Illustrating the clusters in a scatter plot
    '''
plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'], y=centers['y'], s=100, c='black', marker = 'P')
sb.scatterplot(data=df_central,x='Sales',y='Profit', hue='Cluster ID',
               palette = 'rocket')
plt.show()

    '''
    Joining the Central region dataframe with its correspondent Region and
    sub-category attributes from the initial dataframe.
    '''
df_final = pd.concat([df_central,
                      df[['Sub-Category', 'State']]],
                     axis=1, join="inner").drop('Region', axis = 1)

    '''
    Filtering Cluster 2 entries for df_final.
    '''
df_clt2 = df_final[
    df_final['Cluster ID'] == 'Cluster 2']

    '''
    Below is represented the Cluster 2 Grouped Bar Plot for Sales and
    Profit by Furniture Sub-Category, and then by State
    '''

    '''
    Cluster 2 by Sub-Category
    '''
ax = df_clt2.groupby(['Sub-Category']
                     ).mean().plot(kind='bar', color = ['b','m'])
plt.xlabel('')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 0)
plt.title('Cluster 2 by Sub-Category', 
          fontweight='bold')
plt.show()

    '''
    Cluster 2 by State
    '''
ax = df_clt2.groupby(['State']).mean().plot(kind='bar',
                                               color = ['b',
                                                        'm'])
plt.xlabel('')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 35)
plt.title('Cluster 2 by State', 
          fontweight='bold')
plt.show()
#%%


#%%
"""" DBSCAN CLUSTERING """



    '''
    We'll now apply DBScan clustering. Let's first copy the initial dataframe 
    and then find the average distances
    '''

df_dbscan = df_final.copy().drop('Cluster ID',
                                 axis = 1)
avg_distances=[]

for i in range (1,5):
    ngb = Nngb(n_neighbors = i).fit(df_dbscan[['Sales','Profit']])
    avg_distances, NearestNeighbors_index = ngb.kneighbors(df_dbscan[
        ['Sales','Profit']])

    '''
    Next we sort the average distances array to plot the data in order to 
    find the elbow point, which represents the ideal radius for our dataset
    '''
    
    '''
    Sorting the average distances array
    '''
sorted_distances = np.sort(avg_distances, axis=0)

    '''
    Plotting the average distances
    '''
plt.plot(sorted_distances[:,3], '-.')
plt.xlabel('Data Points')
plt.ylabel('Average Distances')
plt.ylim(0,100)
plt.show()

    '''
    DBScan clustering for optimal radius and min_samples = 4 (two times the
    dimension).
    '''
DBSCAN_clustering = DBSCAN(eps = 70,
                          min_samples = 4).fit(df_dbscan[['Sales','Profit']])

    '''
    Data labelling according to the identified clusters
    '''
df_dbscan['Cluster ID'] = DBSCAN_clustering.labels_  

    '''
    Below we check the Labels of each cluster and the outlier's. And next is
    counted the amount of records of each group
    '''
labels = np.unique(DBSCAN_clustering.labels_)

Outliers = np.count_nonzero(labels<0)

n_clusters_dbscan = labels.size-Outliers

df_dbscan['Cluster ID'].value_counts()

    '''
    Scatter plot for the identified clusters
    '''
fig, (axes) = plt.subplots(figsize=(10,6))

sb.scatterplot(x = 'Sales', y = 'Profit', 
               data = df_dbscan[df_dbscan['Cluster ID']!=-1],
               hue='Cluster ID', palette='Paired', ax=axes, s = 100)
outliers = df_dbscan [df_dbscan['Cluster ID']==-1]
axes.scatter(outliers['Sales'], outliers['Profit'], label='Outliers',
             marker ='o', color='m')
axes.legend()
plt.title('Identified Clusters', fontsize = 18,
          fontweight = 'bold')
plt.show()


    '''
    Creating dataframe for outliers
    '''
df_dbscan_outlier = df_dbscan[df_dbscan['Cluster ID'] == -1].drop('Cluster ID',
                                                                axis = 1)


    '''
    Filtering by State
    '''
ax = df_dbscan_outlier.groupby(['State']).mean().plot(kind='bar',
                                               color = ['b',
                                                        'm'])
plt.xlabel('')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 35)
plt.title('Outliers by State',
          fontweight='bold')
plt.show()


    '''
    Filtering by Sub-Category
    '''

ax =  df_dbscan_outlier.groupby(['Sub-Category']).mean().plot(kind='bar',
                                               color = ['b',
                                                        'm'])
plt.xlabel('')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 0)
plt.title('Outliers by Sub-Category',
          fontweight='bold')
plt.show()
___________________________________END_________________________________________