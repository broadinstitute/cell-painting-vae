from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow import keras
from vae_utils import connect_encoder, connect_decoder, LossCallback

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
        lam=0,
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
        self.lam = K.variable(lam)
        self.loss = loss
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.encoder_batch_norm = encoder_batch_norm
        self.verbose = verbose
    

    def compile_loss(self):
        
        def compute_kernel(x, y):
            x_size = K.shape(x)[0]
            y_size = K.shape(y)[0]
            dim = K.shape(x)[1]
            tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
            tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
            return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

        def compute_mmd(x, y):
            x_kernel = compute_kernel(x, x)
            y_kernel = compute_kernel(y, y)
            xy_kernel = compute_kernel(x, y)
            return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
        
       
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
        
        batch_size = K.shape(self.encoder_block["z"])[0]
        latent_dim = K.int_shape(self.encoder_block["z"])[1]
        true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
        self.mmd_loss = compute_mmd(true_samples, self.encoder_block["z"])
        
        mmd_loss = K.get_value(self.lam)* self.mmd_loss
        kl_loss = K.get_value(self.beta)* self.kl_loss
        total_loss = self.reconstruction_loss + mmd_loss + kl_loss
 
        return {
            "loss": total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "kl_loss": kl_loss,
            "mmd_loss": mmd_loss,
        }
    
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

        self.vae_loss = self.compile_loss()['loss']
        kl_loss = self.compile_loss()['kl_loss']
        mmd_loss = self.compile_loss()['mmd_loss']
        reconstruction_loss = self.compile_loss()['reconstruction_loss']
        self.vae.add_loss(self.vae_loss)
        self.vae.add_metric(reconstruction_loss, aggregation='mean', name='recon')
        self.vae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.vae.add_metric(mmd_loss, aggregation='mean', name='mmd')
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
