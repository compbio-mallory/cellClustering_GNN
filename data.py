import numpy as np
import scipy.sparse
from scipy.sparse import base
from scipy import sparse

def load_adjacency_matrix(fname):
    """
    Loads an adjacency matrix from a TSV file.

    Args:
      fname: Path to the TSV file containing the adjacency matrix.

    Returns:
      A numpy array representing the adjacency matrix.
    """

    # Read the TSV file using numpy loadtxt
    data = np.loadtxt(fname, delimiter='\t')

    # Check if the input is square (adjacency matrix)
    if data.shape[0] != data.shape[1]:
      raise ValueError("Input file does not represent a square matrix.")
    data = scipy.sparse.csr_matrix(data)
    return data


def load_tsv(fname, verbose=True):
    """
    Given a TSV file where each row represents a feature and each column represents a node,
    constructs a numpy array.
    """

    # Load the TSV file into a 2D numpy array
    features = np.loadtxt(fname, delimiter='\t')
    features_transpose = features.T
    sparse_features = scipy.sparse.csr_matrix(features_transpose)
    if verbose:
        print(
            "Input matrix has {} features and {} nodes.".format(
                features.shape[0], features.shape[1]
            )
        )

    return sparse_features