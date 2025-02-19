# CLUSTERIGN WITHOUT IMPUTATION BUT WITH THE CONVERSION OF 3s TO 0s.
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import v_measure_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings to make the output cleaner
warnings.filterwarnings("ignore")


def load_data(file_path):
    """
    Load mutation data from the file and preprocess it.
    
    Args:
        file_path (str): Path to the mutation data file.
        
    Returns:
        pd.DataFrame: Preprocessed mutation data as a pandas DataFrame.
    """
    # Load the file with tab-separated values
    data = pd.read_csv(file_path, header=None, sep='\t')
    # Convert all values to numeric, replacing invalid entries with NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    # Replace NaN values with 0 and cast all values to integers
    data = data.fillna(0).astype(int)
    # Replace '3's with '0's as per preprocessing requirement
    data[data == 3] = 0
    return data


def load_true_labels(file_path):
    """
    Load ground truth labels from the cell groups file.
    
    Args:
        file_path (str): Path to the ground truth labels file.
        
    Returns:
        np.ndarray: Array of ground truth cell cluster labels.
    """
    # Load the file with tab-separated values
    labels = pd.read_csv(file_path, sep='\t')
    # Return the 'group' column as an array
    return labels['group'].values


def calculate_mutation_similarity(data):
    """
    Calculate the mutation similarity matrix based on shared occurrences.
    
    Args:
        data (np.ndarray): Mutation data matrix.
        
    Returns:
        np.ndarray: Mutation similarity matrix.
    """
    # Convert the data to a binary matrix (1 for mutation present)
    binary_data = (data == 1).astype(int)
    # Compute the dot product to count shared occurrences of 1's
    similarity_matrix = np.dot(binary_data.T, binary_data)
    return similarity_matrix


def mutation_clustering_spectral(similarity_matrix, num_clusters):
    """
    Perform mutation clustering using Spectral Clustering.
    
    Args:
        similarity_matrix (np.ndarray): Precomputed mutation similarity matrix.
        num_clusters (int): Number of clusters to form.
        
    Returns:
        np.ndarray: Array of cluster labels for each mutation.
    """
    # Initialize spectral clustering with the similarity matrix
    clustering = SpectralClustering(
        n_clusters=num_clusters, affinity='precomputed', random_state=42
    )
    # Fit the model and predict mutation cluster labels
    mutation_labels = clustering.fit_predict(similarity_matrix)
    return mutation_labels


def calculate_cell_similarity(data, mutation_labels):
    """
    Calculate cell similarity matrix based on mutation clustering results.
    
    Args:
        data (np.ndarray): Mutation data matrix.
        mutation_labels (np.ndarray): Cluster labels for mutations.
        
    Returns:
        np.ndarray: Cell similarity matrix based on clustered mutation groups.
    """
    # Initialize a matrix to aggregate mutation clusters for each cell
    clustered_data = np.zeros((data.shape[0], np.max(mutation_labels) + 1))
    for mutation_idx, cluster_id in enumerate(mutation_labels):
        # Add mutation contributions to the respective cluster column
        clustered_data[:, cluster_id] += data[:, mutation_idx]
    # Compute cosine similarity between cells
    similarity_matrix = cosine_similarity(clustered_data)
    return similarity_matrix


def cell_clustering_spectral(similarity_matrix, num_clusters):
    """
    Perform cell clustering using Spectral Clustering.
    
    Args:
        similarity_matrix (np.ndarray): Precomputed cell similarity matrix.
        num_clusters (int): Number of clusters to form.
        
    Returns:
        np.ndarray: Array of cluster labels for each cell.
    """
    # Initialize spectral clustering with the similarity matrix
    clustering = SpectralClustering(
        n_clusters=num_clusters, affinity='precomputed', random_state=42
    )
    # Fit the model and predict cell cluster labels
    cell_labels = clustering.fit_predict(similarity_matrix)
    return cell_labels


def calculate_v_measure(true_labels, predicted_labels):
    """
    Calculate V-Measure to evaluate clustering performance.
    
    Args:
        true_labels (np.ndarray): Ground truth labels.
        predicted_labels (np.ndarray): Predicted cluster labels.
        
    Returns:
        float: V-Measure score.
    """
    # Compute the V-Measure score
    v_measure = v_measure_score(true_labels, predicted_labels)
    return v_measure


def main():
    """
    Main function to perform mutation and cell clustering, and evaluate V-Measure scores.
    """
    # File paths
    mutation_file = "input_genotype.tsv"  # Path to the mutation data file
    ground_truth_file = "cells_groups.tsv"  # Path to the ground truth labels file

    # Load mutation data
    print("Loading mutation data.")
    data = load_data(mutation_file).values
    print(f"Loaded data with shape: {data.shape}")

    # Load true cell labels
    print("Loading true cell labels.")
    true_cell_labels = load_true_labels(ground_truth_file)
    print(f"Loaded true cell labels with {len(np.unique(true_cell_labels))} unique clusters.")

    # Determine the number of mutation clusters
    num_mutation_clusters = len(np.unique(true_cell_labels))
    print(f"Determined number of mutation clusters: {num_mutation_clusters}")

    # Calculate mutation similarity matrix
    print("Calculating mutation similarity matrix.")
    mutation_similarity_matrix = calculate_mutation_similarity(data)
    print(f"Calculated mutation similarity matrix with shape: {mutation_similarity_matrix.shape}")

    # Perform mutation clustering
    print("Performing mutation clustering.")
    predicted_mutation_labels = mutation_clustering_spectral(mutation_similarity_matrix, num_mutation_clusters)

    # Calculate V-Measure for mutation clustering
    print("Calculating V-Measure for mutation clustering.")
    true_mutation_labels = np.repeat(
        np.arange(num_mutation_clusters), data.shape[1] // num_mutation_clusters
    )
    if len(true_mutation_labels) < data.shape[1]:
        true_mutation_labels = np.pad(
            true_mutation_labels, (0, data.shape[1] - len(true_mutation_labels)), constant_values=-1
        )
    v_measure_mutation = calculate_v_measure(true_mutation_labels, predicted_mutation_labels)
    print(f"V-Measure for Mutation Clustering: {v_measure_mutation:.4f}")

    # Calculate cell similarity matrix
    print("Calculating cell similarity matrix.")
    cell_similarity_matrix = calculate_cell_similarity(data, predicted_mutation_labels)

    # Perform cell clustering
    print("Performing cell clustering.")
    predicted_cell_labels = cell_clustering_spectral(cell_similarity_matrix, len(np.unique(true_cell_labels)))

    # Calculate V-Measure for cell clustering
    print("Calculating V-Measure for cell clustering.")
    v_measure_cell = calculate_v_measure(true_cell_labels, predicted_cell_labels)
    print(f"V-Measure for Cell Clustering: {v_measure_cell:.4f}")

    # Save the cell similarity matrix to a file
    np.savetxt("cell_similarity.tsv", cell_similarity_matrix, delimiter="\t")
    print("Saved cell similarity matrix to 'cell_similarity.tsv'")

if __name__ == "__main__":
    main()
