import os
import pandas as pd

import numpy as np
from scipy.spatial.distance import pdist, squareform
import metrics
from sklearn.cluster import KMeans

def find_reads_files(root_dir):
    reads_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "reads.tsv" or file.endswith("reads.tsv"):
                full_path = os.path.join(dirpath, file)
                reads_files.append(full_path)
    return reads_files

# Example usage
root_d = './cellClustering_GNN/simulation'
reads_files = find_reads_files(root_d)

results = []

# Print the found files
# print(f"Found {len(reads_files)} 'reads.tsv' files:")
for f in reads_files:
    reads_df = pd.read_csv(f, sep='\t')
    # print(reads_df)

    # Extract read count matrix (ignoring genomic coordinates for clustering purposes)
    read_counts = reads_df.iloc[:, 3:]

    # Extract cell identifiers (column headers)
    cell_ids = read_counts.columns.tolist()

    # Extract clusters from cell IDs
    cell_clusters = [cell_id.split('_')[0] for cell_id in cell_ids]

    # Display basic stats for verification
    read_counts_summary = {
        "Shape": read_counts.shape,
        "Example cell IDs": cell_ids[:5],
        "Unique clusters": set(cell_clusters)
    }

    print(read_counts_summary)



    # Calculate pairwise L1 (Manhattan) distances between cells
    read_count_matrix = read_counts.to_numpy()
    l1_distances = squareform(pdist(read_count_matrix.T, metric='cityblock'))

    # Convert distance matrix to a DataFrame for clarity
    l1_distance_df = pd.DataFrame(l1_distances, index=cell_ids, columns=cell_ids)

    # Display a portion of the distance matrix
    print(l1_distance_df.shape)



    # Number of clusters (based on the unique clusters in the data)
    n_clusters = len(set(cell_clusters))

    # Transpose read count matrix to shape (cells × features)
    cell_features = read_count_matrix.T

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(cell_features)

    # Create a DataFrame to compare true and predicted labels
    clustering_results = pd.DataFrame({
        "Cell ID": cell_ids,
        "True Cluster": cell_clusters,
        "Predicted Cluster": predicted_labels
    })

    # print(clustering_results)

    print(f)
    print(metrics.v_measure(clustering_results['True Cluster'], clustering_results['Predicted Cluster']))
    temp_r = {"path": f.split("/")[-3] + "/" + f.split("/")[-2], "rep": f.split("/")[-2], "V_measure": metrics.v_measure(clustering_results['True Cluster'], clustering_results['Predicted Cluster']), "elapsed_time": 0}
    results.append(temp_r)


results_df = pd.DataFrame(results)

# Save results to a CSV file
output_file = "./cellClustering_GNN/kmeans_infer_results/merged_v_measre.tsv"
results_df.to_csv(output_file, index=False, sep='\t')

print(f"Results saved to {output_file}")