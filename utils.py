
"""Helper functions for graph processing."""
import numpy as np
import scipy.sparse
from scipy.sparse import base
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import kneighbors_graph

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


def construct_knn_graph(data, k = 15, symmetrize = True):
  graph = kneighbors_graph(data, k)
  if symmetrize:
    graph = graph + graph.T
    graph.data = np.ones(graph.data.shape)
  return graph