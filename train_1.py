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
from tqdm import tqdm
import os

"""
relax fixed seeds
"""
# Set the random seed for TensorFlow operations
seed = random.randint(1,100)
# seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

from data import load_tsv, load_adjacency_matrix
from GATCN_AE import GATCN_AE
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
flags.DEFINE_list(
    'architecture',
    [16],
    'Network architecture in the format `a,b,c,d`.')
flags.DEFINE_float(
    'collapse_regularization',
    1,
    'Collapse regularization.',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate',
    0,
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

def adj_thresholed(adj, threshold):
  # make the values of adj (n, n) 0 if they are below threshold
  adj[adj < threshold] = 0
  adj[adj >= threshold] = 1
  return adj

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

def get_affinity(data, num_cluster = 4, ):
  from sklearn.neighbors import kneighbors_graph
  data = data.todense()
  # convert data to numpy array
  data = np.array(data)
  A = kneighbors_graph(data, num_cluster, mode='connectivity', include_self=False)
  A.toarray()

  return A

def get_matching(data):
  # check if the data is numpy array
  if not isinstance(data, np.ndarray):
    data = data.todense()

  # Ensure the data is binary (convert anything that's not 1 into 0)
  data = np.where(data == 1, 1, 0)

  # Compute the pairwise count of matching ones between SNVs using matrix multiplication (dot product)
  matching_matrix = np.dot(data, data.T)
  
  # get the maximum values for each row
  max_values = matching_matrix.max(axis=1)

  # fill the diagonal with the max values
  np.fill_diagonal(matching_matrix, max_values)

  # normalize the matching matrix row-wise
  matching_matrix_norm = matching_matrix / np.linalg.norm(matching_matrix, axis=1, keepdims=True)

  # make the nan values 0
  matching_matrix_norm = np.nan_to_num(matching_matrix_norm)

  return matching_matrix_norm, matching_matrix

def impute_data(data):
    # Replace '3' with np.nan
    data = np.where(data == 3, np.nan, data)

    # Initialize the KNN imputer
    imputer = KNNImputer(n_neighbors=10)  # You can adjust the number of neighbors

    # Fit and transform the data
    imputed_data = imputer.fit_transform(data)
    
    return imputed_data

def build_dmon(input_features,
               input_graph,
               input_adjacency):
  
  output = input_features
  # GATCN_AE function call 

  hidden_dims = [output.shape[1], 99, 99, FLAGS.architecture]

  print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
  with tf.device('/gpu:2'):
  # Build and train your model here
    output, emb, output_ae = GATCN_AE(hidden_dims)([output, input_graph, input_adjacency])
    pool, pool_assignment = dmon.DMoN(
        FLAGS.n_clusters,
        collapse_regularization=FLAGS.collapse_regularization,
        dropout_rate=FLAGS.dropout_rate)([output, input_adjacency, input_features, output_ae])
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Load and process the data (convert node features to dense, normalize the
  # graph, convert it to Tensorflow sparse tensor.
  if FLAGS.data_path is not None:
    FLAGS.CNA_path = os.path.join(FLAGS.data_path, 'cell_adj_cosine.tsv')
    FLAGS.SNV_path = os.path.join(FLAGS.data_path, 'cell_similarity.tsv')  # use precomputed cell similarity
    FLAGS.labels_path = os.path.join(FLAGS.data_path, 'cells_groups.tsv')
    
  if FLAGS.labels_path is not None:
    true_labels, _ = metrics.truth_values(FLAGS.labels_path)

  start = time.time()
  # Load CNA cosine similarity matrix as feature matrix
  CNA_cosine_sparse = load_adjacency_matrix(FLAGS.CNA_path)
  CNA_cosine_dense = CNA_cosine_sparse.todense()
  CNA_cosine = tf.convert_to_tensor(CNA_cosine_dense, dtype=tf.float32)
  # adjacency = utils.construct_knn_graph(CNA_cosine_sparse, k=7)
  
  # Load SNV data
  # SNV_data = load_tsv(FLAGS.SNV_path)
#   SNV_data = np.loadtxt(FLAGS.SNV_path, delimiter="\t")
#   if SNV_data.shape[0] != CNA_cosine.shape[0]:
#     SNV_data = SNV_data.T
  
  print("Loading precomputed cell similarity matrix...")
  SNV_adj = np.loadtxt(FLAGS.SNV_path, delimiter="\t") # directly load matrix
  
  expected_shape = (CNA_cosine.shape[0], CNA_cosine.shape[0])

  if SNV_adj.shape != expected_shape:
        raise ValueError(f"Shape mismatch - {SNV_adj.shape}")

    
  # Impute the SNV data
  """for snv dotproduct test setting only"""
  # SNV_data = impute_data(SNV_data)

  # SNV_adj = cosine_similarity(SNV_data)

  # Replace "3" with "0"
  # SNV_data[SNV_data == 3] = 0

  # Calculate dot product
  # SNV_adj = np.dot(SNV_data, SNV_data.T)
  """for snv dotproduct test setting only"""
  # SNV_adj_dense = SNV_adj
  
  # SNV_data, matching_matrix = get_matching(SNV_data)
  # matching_matrix = sparse.csr_matrix(matching_matrix)
  
  # SNV_adj = utils.construct_knn_graph(features, k=7)
  SNV_adj = sparse.csr_matrix(SNV_adj)

  n_nodes = CNA_cosine.shape[0]
  feature_size = CNA_cosine.shape[1]
  
  # adjacency = CNA_cosine_sparse
  # features = tf.convert_to_tensor(SNV_adj_dense, dtype=tf.float32)
  
  adjacency = SNV_adj
  """for snv =0 setting only"""
  # adjacency.data[:] = 0.001
  # print(adjacency)
  """for snv =0 setting only"""
  features = CNA_cosine.numpy()

  # print(adjacency)
  # Create graph from SNV_adj
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(
      utils.normalize_graph(adjacency.copy()))
  # print(graph)
  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency)
  
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
      # optimizer = tf.keras.optimizers.Adam(lr)
      optimizer = tf.keras.optimizers.AdamW(
          learning_rate=lr,
          weight_decay=1e-6,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-7,
          amsgrad=True,
          name='AdamW'
      )

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

      print(FLAGS.n_clusters)
      for epoch in tqdm(range(FLAGS.n_epochs), desc="Training Progress", unit="epoch"):
      # for epoch in range(FLAGS.n_epochs):
          # print(f"Epoch {epoch + 1}/{FLAGS.n_epochs} is running...")

          loss_values, grads = grad(model, [features, graph_normalized, graph])
          optimizer.apply_gradients(zip(grads, model.trainable_variables)) # back propagation
          if epoch % 10 == 0:
            features_pooled, assignments = model([features, graph_normalized, graph], training=False)
            # perform the gmm clustering on the pooled features
            # print(FLAGS.n_clusters)
            gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features_pooled)
            assignments = gmm.predict(features_pooled)
            
            if FLAGS.labels_path is not None:
              v_measure = metrics.v_measure(true_labels, assignments)
              v_measure_scores.append(v_measure)
              if v_measure > best_v_measure:
                best_v_measure = v_measure
                best_cluster_v_measure = assignments
                best_epoch_v_measure = epoch
            
            unique_labels = np.unique(assignments)
            # print(f"Unique labels: {unique_labels}")
            if len(unique_labels) == 1:
              assignments[-1] = -1
            silhouette_feat_pooled = silhouette_score(features_pooled, assignments)
            # convert feature to numpy array
            feat = features.numpy()
            silhouette_feat = silhouette_score(feat, assignments)
            silhouette_avg = (0.1*silhouette_feat_pooled + 0.1*silhouette_feat)
            
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
            
          # if epoch % 100 == 0:
          #   print(f'epoch {epoch}, losses: ' +
          #         ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
          #   # Obtain the cluster assignments.
          #   features_pooled, assignments = model([CNA_cosine, graph_normalized, graph], training=False)
          #   # perform the gmm clustering on the pooled features
          #   gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features_pooled)
          #   assignments = gmm.predict(features_pooled)
          #   print("clusters: ", assignments)
          #   if FLAGS.labels_path is not None:
          #     print("v_measure: ", metrics.v_measure(true_labels, assignments))

  print("v_measure\t", metrics.v_measure(true_labels, best_cluster_silhouette),"\t",best_v_measure)
  
          
  # Obtain the cluster assignments.
  features, assignments = model([CNA_cosine, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  clusters_str = ', '.join(map(str, clusters))  # Convert elements to string and join with comma.
#   print("clusters: ", clusters_str)
#   # Prints some metrics used in the paper.
#   print('Conductance:', metrics.conductance(CNA_cosine, clusters))
#   print('Modularity:', metrics.modularity(CNA_cosine, clusters))

#   # Print number of nodes in each cluster
#   for i in range(FLAGS.n_clusters):
#     print(f'Cluster {i}: {np.sum(clusters == i)} nodes')

  predicted_clusters = np.array(clusters)


  
  # print(f"V-measure: ", metrics.v_measure(true_labels, predicted_clusters))
  # perform the gmm clustering on the features
  gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features)
  assignments = gmm.predict(features)
  # print("clusters: ", assignments)
  predicted_clusters = assignments
  # Calculate V-measure for the new predicted clusters
#   if FLAGS.labels_path is not None:
#     print(f"V-measure for gmm clustering: ", metrics.v_measure(true_labels, predicted_clusters))
  end = time.time()
  
  print(f"Time taken: {end - start} seconds")
  # print missclassified nodes
  # pred_unique_labels = pd.factorize(predicted_clusters)[0]
  # print("unique labels: ", pred_unique_labels)
  # missclassified = np.where(true_labels != pred_unique_labels)
  # print("Missclassified nodes: ", missclassified)

if __name__ == '__main__':
  app.run(main)
