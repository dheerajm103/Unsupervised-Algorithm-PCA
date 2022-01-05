import pandas as pd                                       # importing libraries
from  matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("C:/Users/dheer/heart disease.csv")      # importing dataset
df
# EDA and Data cleansing*************************************************************************************


duplicates = df.duplicated().sum()                         # checking for duplicates
df1 = df.drop_duplicates()
df1
df1.describe()                                             # describing the data set
df1.info()                                                 # checking for null values and data types

plt.boxplot(df1)                                           # plotting boxplot for outliers
# fplotting scatter plot for bivariate anlysis

plt.scatter(df1.age,df1.thalach);plt.xlabel("age");plt.ylabel("thalach")
plt.scatter(df1.trestbps,df1.chol);plt.xlabel("tretbps");plt.ylabel("chol")
plt.scatter(df1.age,df1.chol);plt.xlabel("age");plt.ylabel("chol")
plt.scatter(df1.age,df1.trestbps);plt.xlabel("age");plt.ylabel("trestbps")

plt.matshow(df1.corr())                                     # plotting matshow for correlation

def norm1(i):                                               # scaling dataset to 0 and 1
    x = (i-i.min())/(i.max()-i.min())
    return x
norm = norm1(df1)

# for Hiererichal clusttering before pca


# calculating istance
z = linkage(norm, method = "complete", metric = "euclidean")
z
# plotting  Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,leaf_font_size = 10 )
plt.show()
# initialising and fitting model for cluster 4
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1['clust'] = cluster_labels 

df2 = df1.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
df2
d1 = df2.iloc[:, 1:].groupby(df2.clust).mean()
d1
# plotting scatter for clusters
plt.scatter(df2.age,df2.thalach, c = h_complete.labels_);plt.title("Heirerichal clusttering before pca")

# Kmeans clusttering before pca

# calculating total within sum of squares
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i).fit(norm)
    TWSS.append(kmeans.inertia_)
TWSS    

# plotting scree plot
plt.plot(k,TWSS,"ro-")  ;plt.xlabel("clusters");plt.ylabel("TWSS");plt.title("scree plot before pca")  

# initialising and fitting model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df1["clust"] = cluster_labels        
df3 = df1.iloc[0:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
df3
d2 = df3.iloc[0:,1:].groupby(df3.clust).mean()
d2
plt.scatter(df3.age,df3.thalach, c = h_complete.labels_);plt.title("kmeans  clusttering before pca")

# PCA FOR Dimension reduction

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(norm)
pca_values
var = pca.explained_variance_ratio_
var
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1
plt.plot(var1, color = "red") ; plt.title("pca variance")
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7", "comp8", "comp9", "comp10", "comp11", "comp12","comp13"
pca_data1 = pca_data.iloc[:,0:4]
pca.components_
pca.components_[0]

# Hiererichal clusttering after pca

# calculating distance
z = linkage(pca_data1, method = "complete", metric = "euclidean")
z
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,leaf_font_size = 10 )
plt.show()
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(pca_data1) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

pca_data1['clust'] = cluster_labels  

pca_data2 = pca_data1.iloc[:, [4,0,1,2,3]]
pca_data2
d3 = pca_data2.iloc[:, 1:].groupby(pca_data1.clust).mean()
d3
plt.scatter(pca_data1.comp0,pca_data1.comp1, c = h_complete.labels_);plt.title("Hiererichal clustering after pca")

# kmeans after pca

# calculating total within sum of square
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i).fit(pca_data2)
    TWSS.append(kmeans.inertia_)
TWSS    

# plotting scree plot
plt.plot(k,TWSS,"ro-")  ;plt.xlabel("clusters");plt.ylabel("TWSS") ;plt.title("Scree plot after pca") 
# initialising and fitting model for cluster 3
model = KMeans(n_clusters = 4).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df1["clust"] = cluster_labels        
pca_data1['clust'] = cluster_labels 

pca_data3 = pca_data1.iloc[:, [4,0,1,2,3]]
pca_data3
d4 = pca_data3.iloc[:, 1:].groupby(pca_data3.clust).mean()
d4
# plotting scatter plot for cluster
plt.scatter(pca_data3.comp0,pca_data3.comp1, c = h_complete.labels_);plt.title("Kmeans clustering after pca")


