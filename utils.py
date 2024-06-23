
"""Helper functions for graph processing."""
import numpy as np
import scipy.sparse
from scipy.sparse import base
import pandas as pd
import tensorflow as tf

def normalize_graph(graph,
                    normalized = True,
                    add_self_loops = True):
  """Normalized the graph's adjacency matrix in the scipy sparse matrix format.

  Args:
    graph: A scipy sparse adjacency matrix of the input graph.
    normalized: If True, uses the normalized Laplacian formulation. Otherwise,
      use the unnormalized Laplacian construction.
    add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
      the graph.

  Returns:
    A scipy sparse matrix containing the normalized version of the input graph.
  """
  if add_self_loops:
    graph = graph + scipy.sparse.identity(graph.shape[0])
  degree = np.squeeze(np.asarray(graph.sum(axis=1)))
  if normalized:
    with np.errstate(divide='ignore'):
      inverse_sqrt_degree = 1. / np.sqrt(degree)
    inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
    inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
    return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
  else:
    with np.errstate(divide='ignore'):
      inverse_degree = 1. / degree
    inverse_degree[inverse_degree == np.inf] = 0
    inverse_degree = scipy.sparse.diags(inverse_degree)
    return inverse_degree @ graph


def get_knn(cosine_similarity, k=6):
  # Mask self-similarities by setting them to a large negative value
  mask = tf.eye(num_rows=tf.shape(cosine_similarity)[0], dtype=tf.bool)
  cosine_similarity_masked = tf.where(mask, -tf.ones_like(cosine_similarity), cosine_similarity)

  # Use TensorFlow's top_k to find the indices of the k-nearest neighbors for each node
  values, indices = tf.math.top_k(cosine_similarity_masked, k=k)

  # Optional: Create a sparse adjacency matrix for the kNN graph
  # This involves creating a tensor of [num_nodes * k, 2] indicating the edges
  num_nodes = tf.shape(cosine_similarity)[0]
  row_indices = tf.range(num_nodes)
  row_indices_repeated = tf.repeat(row_indices, k)

  edges = tf.stack([row_indices_repeated, tf.reshape(indices, [-1])], axis=1)
  edge_weights = tf.reshape(values, [-1])

  # Create a TensorFlow SparseTensor representing the adjacency matrix
  adjacency_matrix = tf.SparseTensor(indices=tf.cast(edges, dtype=tf.int64),
                                     values=edge_weights,
                                     dense_shape=[num_nodes, num_nodes])

  # Optional: Convert the SparseTensor to a dense tensor, if needed
  #adjacency_matrix_dense = tf.sparse.to_dense(adjacency_matrix)
  return adjacency_matrix


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def construct_knn_graph(data, k):
  """
  Construct a k-nearest neighbor graph from data.
  Parameters:
  - data: numpy array of shape (n_samples, n_features), The input data.
  - k: int, The number of nearest neighbors to find for each data point.
  Returns:
  - adjacency_matrix: scipy.sparse.csr_matrix, The adjacency matrix of the kNN graph.
  """
  # Initialize the NearestNeighbors model
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
  
  # Find the k nearest neighbors for each data point (including itself)
  # distances: Array representing the lengths to points, shape (n_samples, k+1)
  # indices: Indices of the nearest points in the population matrix, shape (n_samples, k+1)
  distances, indices = nbrs.kneighbors(data)
  
  # Initialize the adjacency matrix with zeros
  n_samples = data.shape[0]
  adjacency_matrix = np.zeros((n_samples, n_samples))
  
  # Fill in the adjacency matrix
  for i in range(n_samples):
      for j in indices[i, 1:]:  # Skip the first neighbor (itself)
          adjacency_matrix[i, j] = 1
          adjacency_matrix[j, i] = 1  # For undirected graph
  
  # Convert to a sparse CSR matrix for efficiency
  adjacency_matrix_sparse = csr_matrix(adjacency_matrix)
  
  return adjacency_matrix_sparse
