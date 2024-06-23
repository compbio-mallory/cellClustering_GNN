"""Deep Modularity Network (DMoN) Keras layer.

Deep Modularity Network (DMoN) layer implementation as presented in
"Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer.
DMoN optimizes modularity clustering objective in a fully unsupervised regime.
"""
from typing import List
from typing import Tuple
import tensorflow.compat.v2 as tf


class DMoN(tf.keras.layers.Layer):
  """Implementation of Deep Modularity Network (DMoN) layer.

  Deep Modularity Network (DMoN) layer implementation as presented in
  "Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer.
  DMoN optimizes modularity clustering objective in a fully unsupervised mode,
  however, this implementation can also be used as a regularizer in a supervised
  graph neural network. Optionally, it does graph unpooling.

  Attributes:
    n_clusters: Number of clusters in the model.
    collapse_regularization: Collapse regularization weight.
    dropout_rate: Dropout rate. Note that the dropout in applied to the
      intermediate representations before the softmax.
    do_unpooling: Parameter controlling whether to perform unpooling of the
      features with respect to their soft clusters. If true, shape of the input
      is preserved.
  """

  def __init__(self,
               n_clusters,
               collapse_regularization = 0.1,
               dropout_rate = 0,
               do_unpooling = False):
    """Initializes the layer with specified parameters."""
    super(DMoN, self).__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.dropout_rate = dropout_rate
    self.do_unpooling = do_unpooling

  def build(self, input_shape):
    """Builds the Keras model according to the input shape."""
    self.transform = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            self.n_clusters,
            kernel_initializer='orthogonal',
            bias_initializer='zeros'),
        tf.keras.layers.Dropout(self.dropout_rate)
    ])
    super(DMoN, self).build(input_shape)

  def call(
      self, inputs):
    """Performs DMoN clustering according to input features and input graph.

    Args:
      inputs: A tuple of Tensorflow tensors. First element is (n*d) node feature
        matrix and the second one is (n*n) sparse graph adjacency matrix.

    Returns:
      A tuple (features, clusters) with (k*d) cluster representations and
      (n*k) cluster assignment matrix, where k is the number of cluster,
      d is the dimensionality of the input, and n is the number of nodes in the
      input graph. If do_unpooling is True, returns (n*d) node representations
      instead of cluster representations.
    """
    features, adjacency, input_features, recon = inputs

    assert isinstance(features, tf.Tensor)
    assert isinstance(adjacency, tf.SparseTensor)
    assert len(features.shape) == 2
    assert len(adjacency.shape) == 2
    assert features.shape[0] == adjacency.shape[0]

    assignments = tf.nn.softmax(self.transform(features), axis=1)
    cluster_sizes = tf.math.reduce_sum(assignments, axis=0)  # Size [k].
    assignments_pooling = assignments / cluster_sizes  # Size [n, k].

    degrees = tf.sparse.reduce_sum(adjacency, axis=0)  # Size [n].
    degrees = tf.reshape(degrees, (-1, 1))

    number_of_nodes = adjacency.shape[1]
    number_of_edges = tf.math.reduce_sum(degrees)

    # Computes the size [k, k] pooled graph as S^T*A*S in two multiplications.
    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(adjacency, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)

    # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
    # in three matrix multiplications by first processing the left part S^T*d
    # and then multyplying it by the right part d^T*S.
    # Left part is [k, 1] tensor.
    normalizer_left = tf.matmul(assignments, degrees, transpose_a=True)
    # Right part is [1, k] tensor.
    normalizer_right = tf.matmul(degrees, assignments, transpose_a=True)

    # Normalizer is rank-1 correction for degree distribution for degrees of the
    # nodes in the original graph, casted to the pooled graph.
    normalizer = tf.matmul(normalizer_left,
                           normalizer_right) / 2 / number_of_edges
    spectral_loss = -tf.linalg.trace(graph_pooled -
                                     normalizer) / 2 / number_of_edges
    self.add_loss(spectral_loss)

    collapse_loss = tf.norm(cluster_sizes) / number_of_nodes * tf.sqrt(
        float(self.n_clusters)) - 1
    self.add_loss(self.collapse_regularization * collapse_loss)

    # Calculate MSE loss between the features and reconstructions
    mse_loss = tf.reduce_mean(tf.square(input_features - recon))
    self.add_loss(mse_loss)
    # Calculate binary cross entropy loss between the features and reconstructions
    # bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_features, recon))
    # self.add_loss(bce_loss)

    features_pooled = tf.matmul(assignments_pooling, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpooling:
      features_pooled = tf.matmul(assignments_pooling, features_pooled)
    return features, assignments
