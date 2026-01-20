# SCGclust: Single Cell Graph clustering

## Introduction
This project implements a graph clustering algorithm for single-cell genomic data using a Graph Attention Convolutional Network Autoencoder (GATCN_AE) combined with Deep Modularity Networks (DMoN). It is particularly suited for clustering cells based on genetic data, such as Copy Number Alterations (CNA) and Single Nucleotide Variants (SNV).

### Workflow
1. **Feature Extraction**: The model computes a feature matrix from SNV data using one of four similarity metrics (dot product, cosine similarity, Euclidean distance, or Pearson correlation).
2. **Graph Construction**: The SNV-derived similarity matrix serves as the graph adjacency matrix, while the CNA cosine similarity matrix provides node features.
3. **Autoencoder**: The model uses a configurable autoencoder (GAT, GCN, or hybrid GATCN) to learn node embeddings that capture the structure of the graph.
4. **Clustering**: DMoN pooling layer performs graph-based clustering by optimizing modularity and other clustering objectives.
5. **Post-processing**: Gaussian Mixture Model (GMM) clustering is applied to the pooled features for final cluster assignments.
6. **Evaluation**: The results are evaluated using metrics including V-measure, Silhouette score, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and modularity.

### Main Components
1. **GATCN_AE**: A graph autoencoder that learns node embeddings using Graph Attention Convolutional Networks (supports GAT, GCN, or hybrid modes).
2. **DMoN**: A clustering layer that groups nodes into clusters by optimizing modularity and other clustering objectives.

## Installation

### Prerequisites
Python 3.7 or higher

TensorFlow 2.x

NumPy

SciPy

scikit-learn

pandas

absl-py

### Steps
```sh
$ git clone https://github.com/compbio-mallory/cellClustering_GNN.git
$ conda create -n <env-name> python=3.11
$ conda activate <env-name>
$ pip install -r requirements.txt
```
## Usage

### Command Line Arguments

- `--CNA_path`: Path to the CNA cosine similarity data file.
- `--SNV_path`: Path to the SNV data file.
- `--labels_path`: Path to the true labels file (default: `data/cell_idx.tsv`).
- `--data_path`: Path to the data directory containing CNA, SNV, and labels files. If provided, individual paths can be omitted.
- `--architecture`: Comma-separated list defining the network architecture hidden layer sizes (default: `99,99,16`).
- `--n_clusters`: Number of clusters to identify (default: 4).
- `--n_epochs`: Number of training epochs (default: 1000).
- `--learning_rates`: Comma-separated list of learning rates to try (default: `0.001`).
- `--dropout_rate`: Dropout rate for the network (default: 0.5, range: 0-1).
- `--collapse_regularization`: Regularization parameter for the clustering layer (default: 1).
- `--model_type`: Type of autoencoder to use: `gat`, `gcn`, or `gatcn` (default: `gatcn`).
- `--feature_type`: Similarity metric for computing feature matrix from SNV data: `dot`, `cosine`, `euclidean`, or `pearson` (default: `euclidean`).

### Example Command
```sh
$ python train.py \
  --CNA_path data/CNA_cosine.tsv \
  --SNV_path data/SNVs.tsv \
  --architecture 99,99,16 \
  --n_clusters 4 \
  --n_epochs 1000 \
  --learning_rates 0.001 \
  --dropout_rate 0.5 \
  --collapse_regularization 1 \
  --model_type gatcn \
  --feature_type euclidean
```

### Alternative: Using Data Path
```sh
$ python train.py \
  --data_path data/ \
  --architecture 99,99,16 \
  --n_clusters 4 \
  --n_epochs 1000 \
  --model_type gatcn \
  --feature_type euclidean
```
## Data Preparation

Your data should be organized as follows:

1. **Data Directory** (default: `data/`)
2. **CNA Cosine Similarity Matrix** (`CNA_cosine.tsv`): A TSV file representing the cosine similarity between cells based on CNA data. Use the `get_cna_cosine_similarity()` function from `data.py` to generate this (modify the function as needed to exclude unwanted columns).
3. **SNV Data** (`SNVs.tsv`): A TSV file containing SNV (Single Nucleotide Variant) genotype data for each cell. Values of "3" are replaced with "0" during processing.
4. **Labels** (`cell_idx.tsv`): A TSV file with true cluster labels for evaluation (optional, used for computing metrics like v-measure, ARI, and NMI).

### Expected File Structure
```
data/
├── CNA_cosine.tsv       # Cosine similarity matrix (N × N)
├── SNVs.tsv             # SNV genotype data (N × M)
└── cell_idx.tsv         # True cluster labels (optional)
```

Where N = number of cells and M = number of SNV features.

## Output and Evaluation Metrics

The training process evaluates clustering performance using the following metrics:

- **V-measure**: Measures the homogeneity and completeness of the clustering.
- **Silhouette Score**: Measures how similar a cell is to its own cluster compared to other clusters.
- **Adjusted Rand Index (ARI)**: Measures the similarity between predicted and true cluster assignments (adjusted for chance).
- **Normalized Mutual Information (NMI)**: Measures the mutual dependence between predicted and true clusters.
- **Modularity**: Measures the strength of the graph clustering structure.

The best clustering is selected based on V-measure score during training.

## Data Simulation

Simulation data can be generated using the following commands:
```
bash simulation/run.log
```

## Citation


## Acknowledgement
This project builds upon the work of others in the field of graph neural networks and graph clustering. We acknowledge the following contributions:

### Graph Clustering with Graph Neural Networks
```
@inproceedings{tsitsulin2020clustering,
     author={Tsitsulin, Anton and Palowitch, John and Perozzi, Bryan and M\"uller, Emmanuel}
     title={Graph Clustering with Graph Neural Networks},
     year = {2020},
    }
```

Usage: The Deep Modularity Networks (DMoN) clustering approach is based on the methods described in this paper. We modified the original code to integrate DMoN into our model for effective graph clustering

### ADEPT: Autoencoder with Differentially Expressed Genes and Imputation for a Robust Spatial Transcriptomics Clustering
```
Y. Hu, Y. Zhao, C. T. Schunk, Y. Ma, T. Derr, X. M. Zhou. ADEPT: autoencoder with
differentially expressed genes and imputation for robust spatial transcriptomics clustering
iScience (2023) 26(6), 106792. (also accepted and presented at RECOMB-Seq, Istanbul, Turkey, April 14-15, 2023)
```

Usage: The implementation of the Graph Attention Convolutional Network Autoencoder (GATCN_AE) is adapted from this work. It provided the foundational architecture for our autoencoder module.
