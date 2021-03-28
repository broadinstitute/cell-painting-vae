from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from vae_utils import connect_encoder, connect_decoder


class VAE:
    def __init__(
        self,
        input_dim,
        latent_dim,
        epochs=50,
        batch_size=128,
        optimizer="adam",
        learning_rate=0.0005,
        epsilon_std=1.0,
        beta=0,
        loss="binary_crossentropy",
        encoder_architecture=[],
        decoder_architecture=[],
        encoder_batch_norm=True,
        verbose=True,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epsilon_std = epsilon_std
        self.beta = K.variable(beta)
        self.loss = loss
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.encoder_batch_norm = encoder_batch_norm
        self.verbose = verbose

    def compile_loss(self):
        if self.loss == "binary_crossentropy":
#             loss_fxn = binary_crossentropy(self.inputs, self.cycle)
            loss_fxn = mse(self.inputs, self.cycle)
        elif self.loss == "mse":
            loss_fxn = mse(self.inputs, self.cycle)

        self.reconstruction_loss = loss_fxn
        self.reconstruction_loss *= self.input_dim

        self.kl_loss = (
            1
            + self.encoder_block["z_log_var"]
            - K.square(self.encoder_block["z_mean"])
            - K.exp(self.encoder_block["z_log_var"])
        )
        self.kl_loss = K.sum(self.kl_loss, axis=-1)
        self.kl_loss *= -0.5
        vae_loss = K.mean(
            self.reconstruction_loss + (K.get_value(self.beta) * self.kl_loss)
        )

        return vae_loss

    def compile_encoder(self, name="encoder"):
        self.encoder_block = connect_encoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            architecture=self.encoder_architecture,
            batch_norm=self.encoder_batch_norm,
        )
        self.inputs = self.encoder_block["inputs"]

    def compile_decoder(self, name="decoder"):
        self.decoder_block = connect_decoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            architecture=self.decoder_architecture,
        )

    def compile_vae(self):
        self.compile_encoder()
        self.compile_decoder()
        self.setup_optimizer()

        # instantiate VAE model
        self.cycle = self.decoder_block["decoder"](
            self.encoder_block["encoder"](self.inputs)[2]
        )
        self.vae = Model(self.inputs, self.cycle, name="vae_mlp")

        self.vae_loss = self.compile_loss()
        self.vae.add_loss(self.vae_loss)
        self.vae.compile(optimizer=self.optim)

    def setup_optimizer(self):
        if self.optimizer == "adam":
            self.optim = optimizers.Adam(lr=self.learning_rate)

    def train(self, x_train, x_test):
        if not hasattr(self, "vae_loss"):
            self.compile_vae()
        # train the autoencoder
        self.vae.fit(
            x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_test, None),
            verbose=self.verbose,
        )
