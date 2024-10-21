# SCGclust: Single Cell Graph clustering

## Introduction
This project implements a graph clustering algorithm using a Graph Attention Convolutional Network Autoencoder (GATCN_AE) and Deep Modularity Networks (DMoN). It is particularly suited for clustering cells based on genetic data, such as Copy Number Alterations (CNA) and Single Nucleotide Variants (SNV).

The main components are:

1.	GATCN_AE: A graph autoencoder that learns node embeddings using Graph Attention Convolutional Networks.
2.	DMoN: A clustering layer that groups nodes into clusters by optimizing modularity and other clustering objectives.

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
$ conda create -n <env-name> python=3.7
$ source activate <env-name>
$ pip install -r requirements.txt
```
## Usage

--CNA_path: Path to the CNA cosine similarity data file.

--SNV_path: Path to the SNV data file.

--labels_path: Path to the true labels file.

--data_path: Path to the data directory containing the above files. If provided,individual paths can be omitted.

--architecture: Comma-separated list defining the network architecture (e.g., 16,3264).

--n_clusters: Number of clusters to identify.

--n_epochs: Number of training epochs.

--learning_rates: Comma-separated list of learning rates to try.

--dropout_rate: Dropout rate for the network (between 0 and 1).

--collapse_regularization: Regularization parameter for the clustering layer.

### Example Command
```sh
$ python train.py \
  --CNA_path data/CNA_cosine.tsv \
  --SNV_path data/SNVs.tsv
  --architecture 16 \
  --n_clusters 4 \
  --n_epochs 1000 \
  --learning_rates 0.001 \
  --dropout_rate 0.5 \
  --collapse_regularization 1
```
## Data Preperation 

Your data should be organized as follows:

1. Data Directory (data/ by default):
2. CNA Cosine Similarity Matrix (cell_adj_cosine.tsv): A TSV file representing the cosine similarity between cells based on CNA data.
3. Use get_cna_cosine_similarity() function from data.py to get the CNA cosine similarity matrix(please modify the function to remove unwanted coulumns from the data)
4. SNV Data (input_genotype.tsv): A TSV file containing SNV data for each cell.
5. Labels (cells_groups.tsv): A TSV file with true cluster labels for evaluation.

## Citation


## Acknowledgement
This project builds upon the work of others in the field of graph neural networks and graph clustering. We acknowledge the following contributions:

### Graph Clustering with Graph Neural Networks
@inproceedings{tsitsulin2020clustering,
     author={Tsitsulin, Anton and Palowitch, John and Perozzi, Bryan and M\"uller, Emmanuel}
     title={Graph Clustering with Graph Neural Networks},
     year = {2020},
    }

Usage: The Deep Modularity Networks (DMoN) clustering approach is based on the methods described in this paper. We modified the original code to integrate DMoN into our model for effective graph clustering

### ADEPT: Autoencoder with Differentially Expressed Genes and Imputation for a Robust Spatial Transcriptomics Clustering
Y. Hu, Y. Zhao, C. T. Schunk, Y. Ma, T. Derr, X. M. Zhou. ADEPT: autoencoder with differentially expressed genes and imputation for robust spatial transcriptomics clustering. iScience (2023) 26(6), 106792. (also accepted and presented at RECOMB-Seq, Istanbul, Turkey, April 14-15, 2023)

Usage: The implementation of the Graph Attention Convolutional Network Autoencoder (GATCN_AE) is adapted from this work. It provided the foundational architecture for our autoencoder module.
