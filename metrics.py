
"""Clustering metric implementation (pairwise and graph-based)."""
from typing import Tuple
import numpy as np
from scipy.sparse import base
from sklearn.metrics import cluster
from sklearn import metrics
import pandas as pd

import pandas as pd
import numpy as np

def truth_values(file):
    if 'cell_idx' in file:
        cell_id = pd.read_csv(file, sep="\t", header=None, names=['id', 'cell'])
        cell_groups = pd.read_csv("data/T10_groups.tsv", sep="\t")
        cells = cell_id.merge(cell_groups, left_on='cell', right_on='cell')
        mapping = {
            'D': 0,
            'H': 1,
            'A1': 2,
            'A2': 3
        }
        cells['label'] = cells['groups'].map(mapping)
        true_labels = cells['label'].to_numpy()
        unique_labels = np.unique(true_labels)
    else:
        data = pd.read_csv(file, sep="\t")
        true_labels, unique_labels = pd.factorize(data['group'])
    return true_labels, unique_labels

def pairwise_precision(y_true, y_pred):
  """Computes pairwise precision of two clusterings.

  Args:
    y_true: An [n] int ground-truth cluster vector.
    y_pred: An [n] int predicted cluster vector.

  Returns:
    Precision value computed from the true/false positives and negatives.
  """
  true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
  """Computes pairwise recall of two clusterings.

  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

  Returns:
    Recall value computed from the true/false positives and negatives.
  """
  true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_negatives)


def pairwise_accuracy(y_true, y_pred):
  """Computes pairwise accuracy of two clusterings.

  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

  Returns:
    Accuracy value computed from the true/false positives and negatives.
  """
  true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
  return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)


def _pairwise_confusion(
    y_true,
    y_pred):
  """Computes pairwise confusion matrix of two clusterings.

  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

  Returns:
    True positive, false positive, true negative, and false negative values.
  """
  contingency = cluster.contingency_matrix(y_true, y_pred)
  same_class_true = np.max(contingency, 1)
  same_class_pred = np.max(contingency, 0)
  diff_class_true = contingency.sum(axis=1) - same_class_true
  diff_class_pred = contingency.sum(axis=0) - same_class_pred
  total = contingency.sum()

  true_positives = (same_class_true * (same_class_true - 1)).sum()
  false_positives = (diff_class_true * same_class_true * 2).sum()
  false_negatives = (diff_class_pred * same_class_pred * 2).sum()
  true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

  return true_positives, false_positives, false_negatives, true_negatives


def modularity(adjacency, clusters):
  """Computes graph modularity.

  Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

  Returns:
    The value of graph modularity.
    https://en.wikipedia.org/wiki/Modularity_(networks)
  """
  degrees = adjacency.sum(axis=0).A1
  n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
  result = 0
  for cluster_id in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
    degrees_submatrix = degrees[cluster_indices]
    result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
  return result / n_edges


def conductance(adjacency, clusters):
  """Computes graph conductance as in Yang & Leskovec (2012).

  Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

  Returns:
    The average conductance value of the graph clusters.
  """
  inter = 0  # Number of inter-cluster edges.
  intra = 0  # Number of intra-cluster edges.
  cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
  for cluster_id in np.unique(clusters):
    cluster_indices[:] = 0
    cluster_indices[np.where(clusters == cluster_id)[0]] = 1
    adj_submatrix = adjacency[cluster_indices, :]
    inter += np.sum(adj_submatrix[:, cluster_indices])
    intra += np.sum(adj_submatrix[:, ~cluster_indices])
  return intra / (inter + intra)

def v_measure(ground_truth_numeric, predicted_clusters) -> float:
  return metrics.v_measure_score(ground_truth_numeric, predicted_clusters)
