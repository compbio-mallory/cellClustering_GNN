import tensorflow as tf
from tensorflow.keras import layers

class GATConv(layers.Layer):
    def __init__(self, n_features, n_channels, heads=1, activation=None, skip_connection=True, _attention=False, negative_slope=0.2, dropout_rate=0.0, _alpha=True, **kwargs):
        super(GATConv, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.heads = heads
        self.skip_connection = skip_connection
        if activation is not None:
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = None
        self._attention = _attention
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        self.attentions = None
        self._alpha = _alpha

        # Initialize weights for the linear transformation of features
        self.kernel = self.add_weight(shape=(n_features, self.heads * self.n_channels),
                                        initializer='glorot_uniform', name='kernel')
        
        # Initialize weights for the attention mechanism
        if self._attention and self._alpha:
            self.attention_kernel_src = self.add_weight(shape=(1, self.heads, self.n_channels),
                                                        initializer='glorot_uniform', name='attention_kernel_src')
            self.attention_kernel_dst = self.add_weight(shape=(1, self.heads, self.n_channels),
                                                        initializer='glorot_uniform', name='attention_kernel_dst')
        
        # Bias is optional, but included here for completeness
        #self.bias = self.add_weight(shape=(self.heads * self.n_channels,), initializer='zeros', name='bias')

    def call(self, inputs, tied_attention=None):
        features, adjacency = inputs

        # Linear Transformation
        features_transformed = tf.matmul(features, self.kernel)
        features_transformed = tf.reshape(features_transformed, (-1, self.heads, self.n_channels))

        if not self._attention:
            # If not applying attention mechanism, return transformed features
            features_transformed = tf.reshape(features_transformed, (-1, self.heads * self.n_channels))
            return self.activation(features_transformed) if self.activation is not None else features_transformed
        
        if tied_attention is None:
            # Computing node level attention coefficients
            alpha_src = tf.reduce_sum(features_transformed * self.attention_kernel_src, axis=-1)
            alpha_dst = tf.reduce_sum(features_transformed * self.attention_kernel_dst, axis=-1)

            # Computing attention coefficients for each edge
            attention_scores = alpha_src + tf.transpose(alpha_dst)
            self.attentions = attention_scores
        else:
            attention_scores = tied_attention
        
        # Applying Adjacency Weights to attention scores
        attention_scores_weighted = attention_scores * adjacency
        attention_scores_weighted = tf.sparse.reorder(attention_scores_weighted)
        # Convert SparseTensor to Tensor
        dense_attention_scores_weighted = tf.sparse.to_dense(attention_scores_weighted)

        # Apply leaky_relu function 
        attention_scores_weighted = tf.nn.leaky_relu(dense_attention_scores_weighted, alpha=self.negative_slope)
        # Applying Softmax to Normalize Attention Scores
        attention_coefficients = tf.nn.softmax(attention_scores_weighted, axis=-1)
        #attention_coefficients = self.edge_softmax(attention_scores_weighted, adjacency)
        # Applying dropout to attention coefficients
        attention_coefficients = tf.nn.dropout(attention_coefficients, rate=self.dropout_rate)

        # Applying attention coefficients to features
        features_transformed = tf.reshape(features_transformed, (-1, self.heads * self.n_channels))
        output_features = tf.matmul(attention_coefficients, features_transformed)
        
        if self.activation is not None:
            output_features = self.activation(output_features)
        
        return output_features
    
    def edge_softmax(self, features, adjacency):
        # Ensure adjacency is a dense tensor for simplicity
        # For sparse tensors, consider using tf.sparse.to_dense or optimized sparse operations
        adjacency_dense = tf.sparse.to_dense(adjacency) if isinstance(adjacency, tf.sparse.SparseTensor) else adjacency

        # Step 1: Compute exponential of the attention scores
        exp_attention_scores = tf.exp(features)

        # Step 2: Sum exponentials per node (assuming adjacency_dense is the adjacency matrix with attention scores)
        sum_exp_attention_scores_per_node = tf.reduce_sum(exp_attention_scores * adjacency_dense, axis=1, keepdims=True)

        # Step 3: Divide by the sum to get edge softmax scores
        softmax_scores = exp_attention_scores / sum_exp_attention_scores_per_node

        return softmax_scores

