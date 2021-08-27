import pathlib
import pandas as pd
import sklearn.preprocessing
from pycytominer.cyto_utils import infer_cp_features

def load_data(data_splits, dataset = "cell painting"):
    data = {}
    for data_split in data_splits:
        if dataset == 'cell painting':
            file = pathlib.Path("../0.preprocessing/data", f"cell_painting_{data_split}LEVEL5.tsv.gz")
            data[data_split] = pd.read_csv(file, sep="\t")
        elif dataset == 'L1000':
            file = pathlib.Path("../0C.preprocessing/data", f"L1000PHASE2noscaling_{data_split}.tsv.gz")
            data[data_split] = pd.read_csv(file, sep="\t")
            
    return data


def transform(df, features="infer", meta_features="infer", operation="zeroone"):
    if features == "infer":
        features = infer_cp_features(df)
    if meta_features == "infer":
        meta_features = infer_cp_features(df, metadata=True)

    feature_df = df.loc[:, features]
    meta_df = df.loc[:, meta_features]

    if operation == "zeroone":
        scaler = sklearn.preprocessing.MinMaxScaler()
    if operation == "-1+1":
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    feature_df = pd.DataFrame(
        scaler.fit_transform(feature_df),
        index=feature_df.index,
        columns=feature_df.columns,
    )
    output_df = pd.concat([meta_df, feature_df], axis="columns")
    return output_df

def infer_L1000_features(population_df, metadata=False):
    features = [
        x
        for x in population_df.columns.tolist()
        if (
            x.isnumeric()
        )
    ]

    if metadata:
        features = [
        x
        for x in population_df.columns.tolist()
        if (
            not x.isnumeric()
        )
    ]
    
    return features
    
