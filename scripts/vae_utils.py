from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer, BatchNormalization, Activation
from tensorflow.keras import metrics


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class LossCallback(Callback):
    def __init__(self, training_data, original_dim, encoder_cbk, decoder_cbk):
        self.training_data = training_data
        self.original_dim = original_dim
        self.encoder_cbk = encoder_cbk
        self.decoder_cbk = decoder_cbk

    def on_train_begin(self, logs={}):
        self.xent_loss = []
        self.kl_loss = []

    def on_epoch_end(self, epoch, logs={}):
        recon = self.decoder_cbk.predict(self.encoder_cbk.predict(self.training_data))
        xent_loss = approx_keras_binary_cross_entropy(
            x=recon, z=self.training_data, p=self.original_dim
        )
        full_loss = logs.get("loss")
        self.xent_loss.append(xent_loss)
        self.kl_loss.append(full_loss - xent_loss)
        return


def connect_encoder(input_dim, latent_dim, architecture=[], batch_norm=True):
    nodes = {}
    nodes["inputs"] = Input(shape=(input_dim,))

    if len(architecture) == 0:
        idx = "inputs"

    for idx in range(0, len(architecture)):
        node_size = architecture[idx]
        if idx == 0:
            nodes[idx] = Dense(node_size, activation="relu")(nodes["inputs"])
        else:
            nodes[idx] = Dense(node_size, activation="relu")(nodes[idx - 1])

    z_mean = Dense(latent_dim, kernel_initializer="glorot_uniform")(nodes[idx])
    z_log_var = Dense(latent_dim, kernel_initializer="glorot_uniform")(nodes[idx])

    if batch_norm:
        z_mean_batchnorm = BatchNormalization()(z_mean)
        nodes["z_mean"] = Activation("relu")(z_mean_batchnorm)

        z_log_var_batchnorm = BatchNormalization()(z_log_var)
        nodes["z_log_var"] = Activation("relu")(z_log_var_batchnorm)
    else:
        nodes["z_mean"] = Activation("relu")(z_mean)
        nodes["z_log_var"] = Activation("relu")(z_log_var)

    nodes["z"] = Lambda(sampling, output_shape=(latent_dim,))(
        [nodes["z_mean"], nodes["z_log_var"]]
    )

    nodes["encoder"] = Model(
        nodes["inputs"], [nodes["z_mean"], nodes["z_log_var"], nodes["z"]]
    )
    return nodes


def connect_decoder(input_dim, latent_dim, architecture=[]):
    nodes = {}
    nodes["inputs"] = Input(shape=(latent_dim,))

    if len(architecture) == 0:
        idx = "inputs"

    for idx in range(0, len(architecture)):
        node_size = architecture[idx]
        if idx == 0:
            nodes[idx] = Dense(
                node_size, activation="relu", kernel_initializer="glorot_uniform"
            )(nodes["inputs"])
        else:
            nodes[idx] = Dense(
                node_size, activation="relu", kernel_initializer="glorot_uniform"
            )(nodes[idx - 1])

    nodes["outputs"] = Dense(input_dim, activation="sigmoid")(nodes[idx])
    nodes["decoder"] = Model(nodes["inputs"], nodes["outputs"])
    return nodes
