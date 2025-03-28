import scanpy as sc
import pandas as pd

# read in full dataset
# get ground truth labels from metadata
counts = sc.read_mtx('data/gene_sorted-matrix.mtx').T
barcodes = pd.read_csv('data/barcodes.tsv', header=None, sep='\t')
genes = pd.read_csv('data/genes.tsv', header=None, sep='\t')
metadata = pd.read_csv('data/metaData_scDevSC.txt', sep='\t', usecols=["NAME", "New_cellType"])

# get e17 rows
mask = barcodes[0].str.startswith('E17')

# subset files for e17 rows
barcodes = barcodes[mask]
counts = counts[mask.values, :]
counts = counts.copy().T
metadata = metadata.iloc[1:].reset_index(drop=True)
metadata = metadata[mask.values]

# export
counts.write("data/gene_sorted_filtered_matrix.h5ad")
barcodes.to_csv("data/barcodes_filtered.tsv", header=False, index=False, sep="\t")
metadata.to_csv("data/ground_truth_labels.tsv", index=False, sep="\t")