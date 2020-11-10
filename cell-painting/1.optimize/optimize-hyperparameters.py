"""
Optimize several hyperparameters using Bayesian Optimization from Keras Tuner
"""

import sys
import pathlib
import numpy as np
import pandas as pd

from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping

from pycytominer.cyto_utils import infer_cp_features
from optimize_utils import HyperVAE, CustomBayesianTuner, get_optimize_args

sys.path.insert(0, "../scripts")
from utils import load_data

# Load command line arguments
args = get_optimize_args()

# Define architecture
if args.architecture == "onelayer":
    encoder_architecture = []
    decoder_architecture = []
if args.architecture == "twolayer":
    encoder_architecture = [100]
    decoder_architecture = [100]
if args.architecture == "threelayer":
    encoder_architecture = [250, 100]
    decoder_architecture = [100, 250]

# Load Data
data_splits = ["train", "test"]
data_dict = load_data(data_splits)

# Prepare data for training
meta_features = infer_cp_features(data_dict["train"], metadata=True)
cp_features = infer_cp_features(data_dict["train"])

train_features_df = data_dict["train"].reindex(cp_features, axis="columns")
train_meta_df = data_dict["train"].reindex(meta_features, axis="columns")

test_features_df = data_dict["test"].reindex(cp_features, axis="columns")
test_meta_df = data_dict["test"].reindex(meta_features, axis="columns")

# Initialize hyper parameter VAE tuning
hypermodel = HyperVAE(
    input_dim=train_features_df.shape[1],
    min_latent_dim=args.min_latent_dim,
    max_latent_dim=args.max_latent_dim,
    min_beta=args.min_beta,
    max_beta=args.max_beta,
    learning_rate=args.learning_rate,
    encoder_batch_norm=True,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
)

tuner = CustomBayesianTuner(
    hypermodel,
    objective="val_loss",
    max_trials=1000,
    directory=args.directory,
    project_name=args.project_name,
    overwrite=args.overwrite,
)

# Search over hyperparameter space to identify optimal combinations
tuner.search(
    train_features_df,
    validation_data=(test_features_df, None),
    callbacks=[EarlyStopping("val_loss", patience=10, min_delta=1)],
    verbose=True,
)
