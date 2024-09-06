from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from autogluon.common.savers import save_pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset
from tabpfn_client import TabPFNRegressor, init
from tqdm import tqdm

init()  # login once, requires an email (login session is saved in your environment)


@dataclass
class CompetitionInfo:
    label = "price"
    id_column = "id"
    eval_metric = "root_mean_squared_error"
    problem_type = "regression"


if __name__ == "__main__":
    comp_info = CompetitionInfo()

    train_data_og = TabularDataset("train.csv")
    test_data = TabularDataset("test.csv")

    # drop ID column, seems not useful as a feature
    train_data = train_data_og.drop(columns=[comp_info.id_column])
    X, y = train_data.drop(columns=[comp_info.label]), train_data[comp_info.label]
    y_pred_test_w_id = test_data[[comp_info.id_column]].copy()
    X_test = test_data.drop(columns=[comp_info.id_column])

    # -- Clean Data with AutoGluon
    # disable text features, seemed to maybe lead to worse results, hard to tell
    _feature_generator_kwargs = {
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    }

    feature_generator = AutoMLPipelineFeatureGenerator(
        **_feature_generator_kwargs,
    )
    X = feature_generator.fit_transform(X=X, y=y)
    X_test = feature_generator.transform(X=X_test)

    # -- Predict with TabPFNV2 API (sending data to the cloud can take a while)
    # -- Subsample and batch over test data
    X_subsampled = X.sample(n=10000, random_state=0)
    y_subsampled = y.loc[X_subsampled.index].values
    X_subsampled = X_subsampled.values
    batch_size = 2000
    predictions = []
    for start in tqdm(list(range(0, len(X_test), batch_size))):
        tabpfn = TabPFNRegressor()
        tabpfn.fit(X_subsampled, y_subsampled)
        batch_pred = tabpfn.predict(X_test.iloc[start : start + batch_size].values)
        predictions.append(batch_pred)

    # Forward pass of the transformer model in the cloud.
    y_pred_test = np.concatenate(predictions)
    y_pred_test_w_id.loc[:, comp_info.label] = y_pred_test

    save_pd.save(path="tabpfn_submission.csv", df=y_pred_test_w_id)
