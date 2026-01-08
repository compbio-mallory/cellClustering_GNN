from typing import Tuple
from absl import app
from absl import flags
import numpy as np
import scipy.sparse
from scipy.sparse import base
import tensorflow as tf
from scipy import sparse
import random
import time
import utils, dmon, metrics
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Set the random seed for TensorFlow operations
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

from data import load_tsv, load_adjacency_matrix
from GATCN_AE import GATCN_AE, GAT_AE, GCN_AE
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'CNA_path',
    None,
    'Input CNA cosine similarity data path.')
flags.DEFINE_string(
    'SNV_path',
    None,
    'Input SNV data path.')
flags.DEFINE_list("architecture", ["99","99","16"], 
                  "Comma-separated list of hidden layer sizes, e.g. '64,32,16'. "
                  "All hidden layers including intermediate ones can be tuned.")
flags.DEFINE_float(
    'collapse_regularization',
    1,
    'Collapse regularization.',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate',
    0.5,
    'Dropout rate for GNN representations.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_integer(
    'n_clusters',
    4,
    'Number of clusters.',
    lower_bound=0)
flags.DEFINE_integer(
    'n_epochs',
    1000,
    'Number of epochs.',
    lower_bound=0)
flags.DEFINE_list(
    'learning_rates',
    [0.001],
    'Learning rates.',)
flags.DEFINE_string(
    'labels_path',
    'data/cell_idx.tsv',
    'Path to the labels file.')
flags.DEFINE_string(
    'data_path',
    None,
    'Path to the data folder.')
flags.DEFINE_enum("model_type", "gatcn", ["gat", "gcn", "gatcn"],
                  "Which autoencoder to use: GAT, GCN, or Hybrid (GAT+GCN).")
flags.DEFINE_enum(
    "feature_type", 
    "euclidean", 
    ["dot", "cosine", "euclidean", "pearson"],
    "Similarity used to compute the (N×N) feature matrix from SNV.")


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
    """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

    Args:
        matrix: A scipy sparse matrix.

    Returns:
        A ternsorflow sparse matrix (rank-2 tensor).
    """
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
        matrix.shape)

def feature_extractor(SNV: np.ndarray, feature_type: str) -> np.ndarray:
    """
    Compute feature matrix from SNV data.
    Options: dot product, cosine similarity, euclidean distance, pearson correlation.
    """
    if feature_type == "dot":
        features = np.dot(SNV, SNV.T)
    elif feature_type == "cosine":
        norms = np.linalg.norm(SNV, axis=1, keepdims=True)
        features = np.dot(SNV, SNV.T) / (norms @ norms.T + 1e-8)
    elif feature_type == "euclidean":
        sq_sum = np.sum(SNV**2, axis=1, keepdims=True)
        features = np.sqrt(sq_sum + sq_sum.T - 2 * np.dot(SNV, SNV.T))
    elif feature_type == "pearson":
        SNV_centered = SNV - SNV.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(SNV_centered, axis=1, keepdims=True)
        features = np.dot(SNV_centered, SNV_centered.T) / (norms @ norms.T + 1e-10)
    else:
        raise ValueError(f"Unknown feature_type {feature_type}")
    return features

def build_autoencoder(model_type: str, in_dim: int, architecture):
    """
    Return an AE model (GAT, GCN, or hybrid).
    `in_dim` is determined by data.
    `architecture` is a list of hidden layer sizes starting from layer 2.
    """
    # Full architecture: prepend input dimension
    hidden_dims = [in_dim] + [int(x) for x in architecture]

    if model_type == "gat":
        return GAT_AE(hidden_dims)
    elif model_type == "gcn":
        return GCN_AE(hidden_dims)
    elif model_type == "gatcn":
        return GATCN_AE(hidden_dims)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

def build_dmon(input_features, input_graph, input_adj, model_type, feature_size, n_nodes):
    """
    Combine AE with DMoN pooling.
    feature_size = data-determined input dimension.
    architecture = tunable hidden layers starting from layer 2.
    """
    arch = [int(x) for x in FLAGS.architecture]  # tunable hidden layers
    autoencoder = build_autoencoder(model_type, feature_size, arch)

    with tf.device('/cpu:0'):
        output, emb, output_ae = autoencoder([input_features, input_graph, input_adj])
        pool, pool_assignment = dmon.DMoN(
            FLAGS.n_clusters,
            collapse_regularization=FLAGS.collapse_regularization,
            dropout_rate=FLAGS.dropout_rate
        )([output, input_adj, input_features, output_ae])

    return tf.keras.Model(inputs=[input_features, input_graph, input_adj],
                          outputs=[pool, pool_assignment])



def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Load and process the data (convert node features to dense, normalize the
    # graph, convert it to Tensorflow sparse tensor.
    if FLAGS.data_path is not None:
        FLAGS.CNA_path = os.path.join(FLAGS.data_path, 'cell_adj_cosine.tsv')
        FLAGS.SNV_path = os.path.join(FLAGS.data_path, 'input_genotype.tsv')
        FLAGS.labels_path = os.path.join(FLAGS.data_path, 'cells_groups.tsv')

    if FLAGS.labels_path is not None:
        true_labels, _ = metrics.truth_values(FLAGS.labels_path)

    start = time.time()
    # Load CNA cosine similarity matrix as feature matrix
    CNA_cosine_sparse = load_adjacency_matrix(FLAGS.CNA_path)
    CNA_cosine_dense = CNA_cosine_sparse.todense()
    CNA_cosine = tf.convert_to_tensor(CNA_cosine_dense, dtype=tf.float32)

    SNV_data = np.loadtxt(FLAGS.SNV_path, delimiter="\t")
    if SNV_data.shape[0] != CNA_cosine.shape[0]:
        SNV_data = SNV_data.T

    # Replace "3" with "0"
    SNV_data[SNV_data == 3] = 0

    # use feature_extractor to compute feature matrix from SNV data
    SNV_adj = feature_extractor(SNV_data, FLAGS.feature_type)
    
    # SNV_adj = cosine_similarity(SNV_data)

    SNV_adj = sparse.csr_matrix(SNV_adj)

    n_nodes = CNA_cosine.shape[0]
    feature_size = CNA_cosine.shape[1]

    # adjacency = CNA_cosine_sparse
    # features = tf.convert_to_tensor(SNV_adj_dense, dtype=tf.float32)

    adjacency = SNV_adj
    features = CNA_cosine

    # Create graph from SNV_adj
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy()))

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    # model = build_dmon(input_features, input_graph, input_adjacency)
    model = build_dmon(input_features, input_graph, input_adjacency,
                       FLAGS.model_type, feature_size, n_nodes)

    # Computes the gradients wrt. the sum of losses, returns a list of them.
    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            # Apply gradient clipping
            clipnorm = 5.0
            clipvalue = 0.5
            # Uncomment one of the following lines depending on whether you want to clip by norm or by value
            grads = [tf.clip_by_norm(g, clipnorm) if g is not None else None for g in grads]
            grads = [tf.clip_by_value(g, -clipvalue, clipvalue) if g is not None else None for g in grads]

        return model.losses, grads

    for lr in FLAGS.learning_rates:
        # optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.000001, clipnorm=5.0, clipvalue=0.5, amsgrad=True, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='AdamW')
        optimizer = tf.keras.optimizers.Adam(lr)
        # optimizer = tf.keras.optimizers.AdamW(
        #     learning_rate=lr,
        #     weight_decay=1e-6,
        #     beta_1=0.9,
        #     beta_2=0.999,
        #     epsilon=1e-7,
        #     amsgrad=True,
        #     name='AdamW'
        # )

        best_v_measure = 0
        best_cluster_v_measure = None
        best_epoch_v_measure = 0
        best_silhouette = -1
        best_epoch_silhouette = 0
        best_cluster_silhouette = None

        # save the v_measure and silhouette scores for each epoch
        v_measure_scores = []
        silhouette_scores = []
        modularity_scores = []
        losses = []

        model.compile(optimizer, None)
        for epoch in range(FLAGS.n_epochs):
            loss_values, grads = grad(model, [features, graph_normalized, graph])
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) # back propagation
            if epoch % 10 == 0:
                features_pooled, assignments = model([features, graph_normalized, graph], training=False)
                # perform the gmm clustering on the pooled features

                gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features_pooled)
                assignments = gmm.predict(features_pooled)

                if FLAGS.labels_path is not None:
                    v_measure = metrics.v_measure(true_labels, assignments)
                    v_measure_scores.append(v_measure)
                    if v_measure > best_v_measure:
                        best_v_measure = v_measure
                        best_cluster_v_measure = assignments
                        best_epoch_v_measure = epoch

                silhouette_feat_pooled = silhouette_score(features_pooled, assignments)
                # convert feature to numpy array
                feat = features.numpy()
                silhouette_feat = silhouette_score(feat, assignments)
                # silhouette_avg = (0.1*silhouette_feat_pooled + 0.1*silhouette_feat)
                silhouette_avg = silhouette_feat_pooled

                silhouette_scores.append(silhouette_avg)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_epoch_silhouette = epoch
                    best_cluster_silhouette = assignments
                loss = sum(loss_values)
                losses.append(-loss)

                # calculate modularity
                modularity = metrics.modularity(SNV_adj.todense(), assignments)
                modularity_scores.append(modularity)

    ARI = metrics.ARI(true_labels, best_cluster_v_measure) if true_labels is not None else None
    NMI = metrics.NMI(true_labels, best_cluster_v_measure) if true_labels is not None else None
    v_measure_gt = metrics.v_measure(true_labels, best_cluster_v_measure) if true_labels is not None else None
    v_measure_silhouette = metrics.v_measure(true_labels, best_cluster_silhouette) if true_labels is not None else None
    print(f"Final Results | V-measure-silhouette: {v_measure_silhouette:.4f} | V-measure GT: {v_measure_gt:.4f} | ARI: {ARI:.4f} | NMI: {NMI:.4f} | Best Silhouette: {best_silhouette:.4f} | Time: {time.time()-start:.2f}s")
    print("Best cluster silhouette:", best_cluster_silhouette, " best cluster v_measure:", best_cluster_v_measure)
if __name__ == '__main__':
    app.run(main)   