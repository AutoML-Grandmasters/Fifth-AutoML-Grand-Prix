from __future__ import annotations

from dataclasses import dataclass

from autogluon.common.savers import save_pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config


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

    # disable text features, seemed to maybe lead to worse results, hard to tell
    _feature_generator_kwargs = {
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    }

    # Disable usage of original features in stack layer, seems to be a bit better, hard to tell
    ag_args_ensemble = {"use_orig_features": False}

    # Use more n_estimators for RF and XT (300 -> 2400), use larger max_bin for XGBoost (seems to improve scores)
    hyperparameters = get_hyperparameter_config("zeroshot")
    for i in range(len(hyperparameters["RF"])):
        hyperparameters["RF"][i]["n_estimators"] = 2400
    for i in range(len(hyperparameters["XT"])):
        hyperparameters["XT"][i]["n_estimators"] = 2400
    for i in range(len(hyperparameters["XGB"])):
        hyperparameters["XGB"][i]["max_bin"] = 8192

    # Exclude KNN and NN_TORCH, seems like they aren't helpful
    # Disable dynamic stacking since stacking seems good based on holdout experiments, no need to check again
    # Extra: Edited AutoGluon code to stratify bagging splits, treating the problem as multiclass. This helped stabilize CV scores.
    predictor = TabularPredictor(
        label=comp_info.label,
        eval_metric=comp_info.eval_metric,
        problem_type=comp_info.problem_type,
    ).fit(
        train_data=train_data,
        presets="best_quality",
        excluded_model_types=["KNN", "NN_TORCH"],
        hyperparameters=hyperparameters,
        dynamic_stacking=False,
        time_limit=3600 * 6,
        ag_args_ensemble=ag_args_ensemble,
        _feature_generator_kwargs=_feature_generator_kwargs,
    )

    y_pred_test = predictor.predict(test_data)
    y_pred_test_w_id = test_data[[comp_info.id_column]].copy()
    y_pred_test_w_id.loc[:, comp_info.label] = y_pred_test

    save_pd.save(path="autogluon_submission.csv", df=y_pred_test_w_id)
