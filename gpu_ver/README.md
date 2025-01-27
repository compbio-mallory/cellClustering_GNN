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
$ conda create -n <env-name> python=3.11
$ conda activate <env-name>
$ pip install -r requirements.txt

$ pip install 'tensorflow[and-cuda]'
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
$ python ./gpu_ver/train.py \
  --CNA_path data/CNA_cosine.tsv \
  --SNV_path data/SNVs.tsv
  --n_clusters 4 \
  --n_epochs 1000 \
  --learning_rates 0.001 \
  --dropout_rate 0.5 \
  --collapse_regularization 1
```


## Citation


