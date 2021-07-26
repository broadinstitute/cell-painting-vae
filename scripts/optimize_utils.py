from tensorflow.keras import backend as K
from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization
import sys
from vae import VAE


class HyperVAE(HyperModel):
    def __init__(
        self,
        input_dim,
        min_latent_dim,
        max_latent_dim,
        epochs=2,
        batch_size=5,
        optimizer="adam",
        learning_rate=0.0005,
        epsilon_std=1.0,
        min_beta=0,
        max_beta=10,
        loss="binary_crossentropy",
        encoder_batch_norm=True,
        encoder_architecture=[100],
        decoder_architecture=[100],
        verbose=True,
    ):
        self.input_dim = input_dim
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epsilon_std = epsilon_std
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.loss = loss
        self.encoder_batch_norm = encoder_batch_norm
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture

    def build(self, hp):
        model = VAE(
            input_dim=self.input_dim,
            latent_dim=hp.Int(
                "latent_dim", self.min_latent_dim, self.max_latent_dim, step=5
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            learning_rate=hp.Choice("learning_rate", values=self.learning_rate),
            epsilon_std=self.epsilon_std,
#             beta=hp.Float("beta", self.min_beta, self.max_beta, step=0.1),
            beta = 1,
            loss=self.loss,
            encoder_batch_norm=hp.Boolean(
                "encoder_batch_norm", default=self.encoder_batch_norm
            ),
            encoder_architecture=self.encoder_architecture,
            decoder_architecture=self.decoder_architecture,
        )
        model.compile_vae()
        return model.vae


class CustomBayesianTunerCellPainting(BayesianOptimization):
    # from https://github.com/keras-team/keras-tuner/issues/122#issuecomment-544648268
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Int("batch_size", 32, 256, step=32)            
        kwargs["epochs"] = trial.hyperparameters.Int("epochs", 10, 15, step=2)
        
        super(CustomBayesianTunerCellPainting, self).run_trial(trial, *args, **kwargs)

class CustomBayesianTunerL1000(BayesianOptimization):
    # from https://github.com/keras-team/keras-tuner/issues/122#issuecomment-544648268
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Int("batch_size", 256, 768, step = 128)
        kwargs["epochs"] = trial.hyperparameters.Int("epochs", 10, 11, step=2)
        
        super(CustomBayesianTunerL1000, self).run_trial(trial, *args, **kwargs)
        

def get_optimize_args():
    """
    Get arguments for the hyperparameter optimization procedure
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="The name of the project")
    parser.add_argument(
        "--directory",
        default="hyperparameter",
        type=str,
        help="The name of the directory to save results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="decision to overwrite already started hyperparameter search",
    )
    parser.add_argument(
        "--min_latent_dim",
        default=10,
        type=int,
        help="Minimum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--max_latent_dim",
        default=100,
        type=int,
        help="Maximum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--min_beta",
        default=0,
        type=int,
        help="Minimum beta penalty applied to VAE KL Divergence",
    )
    parser.add_argument(
        "--max_beta",
        default=5,
        type=int,
        help="Maximum beta penalty applied to VAE KL Divergence",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        nargs="+",
        help="learning rates to use in hyperparameter sweep",
    )
    parser.add_argument("--architecture", default="onelayer", help="VAE architecture")
    parser.add_argument("--dataset", default="cell-painting", help="cell-painting or L1000 dataset")
    args = parser.parse_args()
    return args
