import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import anndata as ad

# Step 1: Load your data (adjust paths to your files)
# gene_expr_df = pd.read_csv('data/e17_expr.csv', index_col=0)  # Gene expression data (genes as rows, cells as columns)
# gene_expr_df = gene_expr_df.T 
# labels_df = pd.read_csv('data/e17_labels.csv', index_col=0)  # Labels data (cells as index)
# print("Expr shape: ", gene_expr_df.shape)
# print("Labels shape: ", labels_df.shape)

# # Step 2: Prepare AnnData object for Scanpy
# adata = sc.AnnData(X=gene_expr_df.values)  # Store gene expression data
# adata.obs['labels'] = labels_df.values.flatten()  # Add labels to adata object

adata = sc.read_mtx('data/gene_sorted-matrix.mtx')
labels = sc.read_csv('data/barcodes.tsv', delimiter='\t')

print("Adata shape: ", adata.shape)
print("Labels shape: ", labels.shape)

1/0

adata.obs['labels'] = labels.values.flatten() 

# Step 3: Preprocessing (optional but recommended)
# Normalize the data (you can adjust this step based on your dataset)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize each cell to have the same total count
sc.pp.log1p(adata)  # Log transform the data

# Step 4: Perform K-means clustering
num_clusters = 3  # You can set the number of clusters based on your problem
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
adata.obs['kmeans_labels'] = kmeans.fit_predict(adata.X)  # Assign KMeans labels to adata

# Step 5: Visualize the clustering (using UMAP for visualization)
sc.pp.neighbors(adata)
sc.tl.umap(adata)  # Run UMAP for dimensionality reduction

# Plot the clustering result (UMAP visualization)
plt.figure(figsize=(8, 6))
sc.pl.umap(adata, color=['kmeans_labels', 'labels'], legend_loc='on data')
plt.show()

# Optionally, you can compare the KMeans labels with the ground truth labels
print("KMeans Clusters:\n", adata.obs['kmeans_labels'].value_counts())
print("True Labels:\n", adata.obs['labels'].value_counts())