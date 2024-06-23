from typing import Tuple
from absl import app
from absl import flags
import numpy as np
import scipy.sparse
from scipy.sparse import base
import tensorflow as tf
from scipy import sparse
import random

import utils, dmon, metrics

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Set the random seed for TensorFlow operations
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

from data import load_tsv, load_adjacency_matrix
from GATCN_AE import GATCN_AE
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'graph_path',
    None,
    'Input graph path.')
flags.DEFINE_string(
    'adj_path',
    None,
    'Input adjacency path.')
flags.DEFINE_string(
    'features_path',
    None,
    'Input features path.')
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

def build_dmon(input_features,
               input_graph,
               input_adjacency):
  
  output = input_features
  # GATCN_AE function call 

  hidden_dims = [output.shape[1], 99, 99, FLAGS.architecture]
  with tf.device('/cpu:0'):
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
    FLAGS.adj_path = os.path.join(FLAGS.data_path, 'cell_adj_cosine.tsv')
    FLAGS.features_path = os.path.join(FLAGS.data_path, 'input_genotype.tsv')
    FLAGS.labels_path = os.path.join(FLAGS.data_path, 'cells_groups.tsv')

  adjacency = load_adjacency_matrix(FLAGS.adj_path)
  adjacency = adjacency.todense()
  #print(adjacency_bin)
  features = load_tsv(FLAGS.features_path)
  features = features.T
  # features_adj = get_affinity(features, 7)
  from sklearn.metrics.pairwise import cosine_similarity
  features_adj = cosine_similarity(features)
  features_adj = sparse.csr_matrix(features_adj)

  n_nodes = adjacency.shape[0]
  feature_size = adjacency.shape[1]

  # Create graph from features_adj
  graph = convert_scipy_sparse_to_sparse_tensor(features_adj)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(
      utils.normalize_graph(features_adj.copy()))

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
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  for lr in FLAGS.learning_rates:
      optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.000001, clipnorm=5.0, clipvalue=0.5, amsgrad=True, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='AdamW')
      #optimizer = tf.keras.optimizers.Adam(lr)
      model.compile(optimizer, None)
      for epoch in range(FLAGS.n_epochs):
          loss_values, grads = grad(model, [adjacency, graph_normalized, graph])
          optimizer.apply_gradients(zip(grads, model.trainable_variables)) # back propagation
          if epoch % 100 == 0:
            print(f'epoch {epoch}, losses: ' +
                  ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
            # Obtain the cluster assignments.
            features_pooled, assignments = model([adjacency, graph_normalized, graph], training=False)
            # perform the gmm clustering on the pooled features
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features_pooled)
            assignments = gmm.predict(features_pooled)
            print("clusters: ", assignments)


          
  # Obtain the cluster assignments.
  features, assignments = model([adjacency, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  clusters_str = ', '.join(map(str, clusters))  # Convert elements to string and join with comma.
  print("clusters: ", clusters_str)
  # Prints some metrics used in the paper.
  print('Conductance:', metrics.conductance(adjacency, clusters))
  print('Modularity:', metrics.modularity(adjacency, clusters))

  # Print number of nodes in each cluster
  for i in range(FLAGS.n_clusters):
    print(f'Cluster {i}: {np.sum(clusters == i)} nodes')

  predicted_clusters = np.array(clusters)


  true_labels = metrics.truth_values(FLAGS.labels_path)

  print(f"V-measure: ", metrics.v_measure(true_labels, predicted_clusters))

  # perform the gmm clustering on the features
  gmm = GaussianMixture(n_components=FLAGS.n_clusters, covariance_type='full').fit(features)
  assignments = gmm.predict(features)
  print("clusters: ", assignments)
  predicted_clusters = assignments
  # Calculate V-measure for the new predicted clusters
  print(f"V-measure for gmm clustering: ", metrics.v_measure(true_labels, predicted_clusters))

  # print missclassified nodes
  print("Missclassified nodes: ", np.where(true_labels != predicted_clusters))
if __name__ == '__main__':
  app.run(main)
