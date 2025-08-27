from typing import Tuple
from absl import app, flags
import numpy as np
import random, time, os
import tensorflow as tf
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import utils, dmon, metrics
from data import load_tsv, load_adjacency_matrix
from GATCN_AE import GATCN_AE, GAT_AE, GCN_AE
# from GAT_AE import GAT_AE   # <-- you’ll need the GAT-only class from before
# from GCN_AE import GCN_AE   # <-- and the GCN-only class

# Reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

FLAGS = flags.FLAGS

# ---------------- FLAGS ----------------
flags.DEFINE_string("data_path", None, "Path to dataset folder")
flags.DEFINE_string("labels_path", None, "Path to true labels (optional)")
flags.DEFINE_enum("model_type", "gatcn", ["gat", "gcn", "gatcn"],
                  "Which autoencoder to use: GAT, GCN, or Hybrid (GAT+GCN).")
flags.DEFINE_list("architecture", ["99","99","16"], 
                  "Comma-separated list of hidden layer sizes, e.g. '64,32,16'. "
                  "All hidden layers including intermediate ones can be tuned.")
flags.DEFINE_integer("n_clusters", 4, "Number of clusters")
flags.DEFINE_integer("n_epochs", 500, "Training epochs")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_float("collapse_regularization", 1.0, "DMoN collapse reg strength")
flags.DEFINE_float("dropout_rate", 0.0, "Dropout for DMoN")

# ---------------- HELPERS ----------------
def convert_scipy_sparse_to_sparse_tensor(matrix: sp.csr_matrix) -> tf.sparse.SparseTensor:
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32), matrix.shape
    )

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

# ---------------- TRAIN LOOP ----------------
def train(model, features, graph_norm, graph, true_labels=None, 
          use_clip=True, clip_norm=5.0, clip_value=0.5, clip_type="norm"):
    """
    Train the model with optional gradient clipping.
    
    Args:
        model: tf.keras.Model
        features: input features
        graph_norm: normalized adjacency or graph input
        graph: adjacency or graph input
        true_labels: optional ground truth labels for metrics
        use_clip: whether to apply gradient clipping
        clip_norm: max norm for clip_by_norm
        clip_value: max absolute value for clip_by_value
        clip_type: 'norm' or 'value', type of clipping to apply
    """
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    best_v, best_sil = 0, -1
    best_epoch_v, best_epoch_s = 0, 0
    best_cluster_v, best_cluster_s = None, None

    def grad_step():
        with tf.GradientTape() as tape:
            _ = model([features, graph_norm, graph], training=True)
            loss = sum(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradient clipping if requested
        if use_clip:
            if clip_type == "norm":
                grads = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in grads]
            elif clip_type == "value":
                grads = [tf.clip_by_value(g, -clip_value, clip_value) if g is not None else None for g in grads]
            else:
                raise ValueError("clip_type must be 'norm' or 'value'")
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(FLAGS.n_epochs):
        loss = grad_step()
        
        if epoch % 10 == 0:
            pooled, _ = model([features, graph_norm, graph], training=False)
            gmm = GaussianMixture(n_components=FLAGS.n_clusters).fit(pooled)
            preds = gmm.predict(pooled)
            
            v = metrics.v_measure(true_labels, preds) if true_labels is not None else None
            s = silhouette_score(pooled, preds)
            
            # Track best V-measure
            if v is not None and v > best_v:
                best_v = v
                best_epoch_v = epoch
                best_cluster_v = preds
            
            # Track best Silhouette
            if s > best_sil:
                best_sil = s
                best_epoch_s = epoch
                best_cluster_s = preds
            
            print(f"[{epoch}] Loss {loss:.4f} | V {v:.3f} | Sil {s:.3f}")

    # Final metrics
    print(f"V-measure at best silhouette (epoch {best_epoch_s}): {metrics.v_measure(true_labels, best_cluster_s)}")
    print(f"Best V-measure (epoch {best_epoch_v}): {best_v}")
    
    return best_v, best_sil, best_cluster_v, best_cluster_s


# ---------------- MAIN ----------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many arguments.")

    # Load data
    CNA = load_adjacency_matrix(FLAGS.data_path)
    CNA_dense = CNA.todense()
    features = tf.convert_to_tensor(CNA_dense, dtype=tf.float32)

    SNV = np.loadtxt(FLAGS.data_path, delimiter="\t")
    if SNV.shape[0] != features.shape[0]:
        SNV = SNV.T
    SNV_adj = sp.csr_matrix(np.dot(SNV, SNV.T))

    graph = convert_scipy_sparse_to_sparse_tensor(SNV_adj)
    graph_norm = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(SNV_adj.copy()))

    n_nodes, feature_size = features.shape
    true_labels, _ = metrics.truth_values(FLAGS.labels_path) if FLAGS.labels_path else (None, None)

    # Build model
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adj = tf.keras.layers.Input((n_nodes,), sparse=True)

    model = build_dmon(input_features, input_graph, input_adj,
                       FLAGS.model_type, feature_size, n_nodes)

    # Train
    start = time.time()
    best_v, best_sil, best_cluster_v, best_cluster_s = train(model, features, graph_norm, graph, true_labels)
    print(f"Finished in {time.time()-start:.2f}s")

if __name__ == "__main__":
    app.run(main)
