import tensorflow as tf
from tensorflow.keras import layers, models
from gat_conv import GATConv
from gcn import GCN

class GATCN_AE(models.Model):
    def __init__(self, hidden_dims):
        super(GATCN_AE, self).__init__()
        in_dim, num_hidden1, num_hidden2, architecture = hidden_dims[0], hidden_dims[1], hidden_dims[2], hidden_dims[3:]
        self.architecture = architecture
        # Initialize GATConv layers
        self.conv1 = GATConv(in_dim, num_hidden1, heads=1, activation='elu', skip_connection=False, _attention=True, _alpha=True)
        self.conv2 = GATConv(num_hidden1, num_hidden2, heads=1, activation=None, skip_connection=False, _attention=False, _alpha=True)

        # Initialize GCN layers based on architecture
        self.gcn_layers = [GCN(n_channels=num_channels, skip_connection=True) for num_channels in architecture]


        self.conv3 = GATConv(num_hidden2, num_hidden1, heads=1, activation='elu', skip_connection=False, _attention=True, _alpha=False)
        self.conv4 = GATConv(num_hidden1, in_dim, heads=1, activation=None, skip_connection=False, _attention=False, _alpha=True)

    def call(self, inputs):
        features, adjacency_normalized, adjacency = inputs
        
        # Forward pass through GATConv layers
        h1 = self.conv1([features, adjacency])
        h2 = self.conv2([h1, adjacency])
        
        # Apply the GCN layer on h2
        # If your adjacency matrix is not a tf.SparseTensor, you will need to convert it
        # adjacency_dense = tf.sparse.to_dense(adjacency) if isinstance(adjacency, tf.SparseTensor) else adjacency
        # Apply the GCN layers on h2
        h2_gcn = h2
        for gcn_layer in self.gcn_layers:
            h2_gcn = gcn_layer([h2_gcn, adjacency_normalized])

        # Sharing weights between conv2 and conv3, and conv1 and conv4
        self.conv3.kernel.assign(tf.transpose(self.conv2.kernel))
        self.conv4.kernel.assign(tf.transpose(self.conv1.kernel))
        
        h3 = self.conv3([h2, adjacency], tied_attention=self.conv1.attentions)
        h4 = self.conv4([h3, adjacency])

        return h2_gcn, h2, h4


# import tensorflow as tf
# from tensorflow.keras import models
# from gat_conv import GATConv

class GAT_AE(models.Model):
    def __init__(self, hidden_dims):
        super(GAT_AE, self).__init__()
        [in_dim, num_hidden1, num_hidden2] = hidden_dims

        # Encoder
        self.conv1 = GATConv(in_dim, num_hidden1, heads=1, activation='elu', skip_connection=False, _attention=True, _alpha=True)
        self.conv2 = GATConv(num_hidden1, num_hidden2, heads=1, activation=None, skip_connection=False, _attention=False, _alpha=True)

        # Decoder (tied weights)
        self.conv3 = GATConv(num_hidden2, num_hidden1, heads=1, activation='elu', skip_connection=False, _attention=True, _alpha=False)
        self.conv4 = GATConv(num_hidden1, in_dim, heads=1, activation=None, skip_connection=False, _attention=False, _alpha=True)

    def call(self, inputs):
        features, adjacency = inputs
        h1 = self.conv1([features, adjacency])
        h2 = self.conv2([h1, adjacency])

        # Tie weights
        self.conv3.kernel.assign(tf.transpose(self.conv2.kernel))
        self.conv4.kernel.assign(tf.transpose(self.conv1.kernel))

        h3 = self.conv3([h2, adjacency], tied_attention=self.conv1.attentions)
        h4 = self.conv4([h3, adjacency])

        return h2, h4  # latent + reconstruction


# import tensorflow as tf
# from tensorflow.keras import models
# from gcn import GCN

class GCN_AE(models.Model):
    def __init__(self, architecture):
        super(GCN_AE, self).__init__()
        # Build encoder and decoder GCNs
        self.encoder_layers = [GCN(n_channels=num_channels, skip_connection=True) for num_channels in architecture]
        self.decoder_layers = [GCN(n_channels=architecture[-2], skip_connection=True),
                               GCN(n_channels=architecture[0], skip_connection=True)]

    def call(self, inputs):
        features, adjacency_normalized = inputs

        # Encoder
        h = features
        for gcn_layer in self.encoder_layers:
            h = gcn_layer([h, adjacency_normalized])
        latent = h

        # Decoder (reverse order)
        h = latent
        for gcn_layer in self.decoder_layers:
            h = gcn_layer([h, adjacency_normalized])
        reconstruction = h

        return latent, reconstruction
