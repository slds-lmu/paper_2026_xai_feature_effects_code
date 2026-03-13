"""
This module contains functions to format dataframes for better readibility and
to simplify plotting.
"""

import pandas as pd
import numpy as np


def format_dataframe(df: pd.DataFrame, bias_squared: bool=False) -> pd.DataFrame:
    """
    Format the dataframe to a more readable format for the table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the bias-variance-decomposition.
    bias_squared : bool
        Whether to use Bias^2 instead of Bias.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    df = df.copy()
    if not bias_squared:
        # bias instead of bias^2
        mask = df["metric"] == "Bias^2"
        df.loc[mask, "value"] = df.loc[mask, "value"] ** 0.5
        df.loc[mask, "metric"] = "Bias"
        bias_name = "Bias"
    else:
        bias_name = "Bias^2"
    df_out = get_grouped_df(df)
    sorted_cols = [
        (feature, metric)
        for feature in sorted(set(col[0] for col in df_out.columns))
        for metric in ["MSE", bias_name, "Variance"]
    ]
    sorted_idx = [
        (n, model, split)
        for n in sorted(df_out.index.get_level_values("sample_size").unique())
        for model in ["LinReg", "GAM_OF", "GAM_OT", "XGBoost_OF", "XGBoost_OT"]
        for split in ["train", "val", "cv"]
    ]
    df_out = df_out.reindex(index=pd.MultiIndex.from_tuples(sorted_idx), columns=sorted_cols)

    return df_out

def format_var_decomp_dataframe(df: pd.DataFrame, multiple_effects: bool=False) -> pd.DataFrame:
    """
    Format the dataframe to a more readable format for the table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the variance decomposition.
    multiple_effects : bool
        Whether the dataframe contains results for multiple effects (e.g., PD and ALE).

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    df = df.copy()
    df_out = get_grouped_df(df, multiple_effects=multiple_effects)
    sorted_cols = [
        (feature, metric)
        for feature in sorted(set(col[0] for col in df_out.columns))
        for metric in ["Variance", "Model Variance", "MC Variance"]
    ]
    if multiple_effects:
        sorted_idx = [
            (effect, n, model, split)
            for effect in ["PD", "ALE"]
            for n in sorted(df_out.index.get_level_values("sample_size").unique())
            for model in ["XGBoost_OF", "XGBoost_OT"]
            for split in ["train", "val", "cv"]
        ]
    else:
        sorted_idx = [
            (n, model, split)
            for n in sorted(df_out.index.get_level_values("sample_size").unique())
            for model in ["XGBoost_OF", "XGBoost_OT"]
            for split in ["train", "val", "cv"]
        ]
    df_out = df_out.reindex(index=pd.MultiIndex.from_tuples(sorted_idx), columns=sorted_cols)

    return df_out


def highlight_min_feature_metric(data: pd.DataFrame) -> pd.DataFrame:
    """
    Highlight the minimum value in each feature-metric pair.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the results of the feature effect analysis.

    Returns
    -------
    pd.DataFrame
        Dataframe with the minimum values highlighted.
    """
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    groups = data.index.droplevel(2).unique()

    for model_n_train in groups:
        group_mask = (data.index.get_level_values(0) == model_n_train[0]) & (
            data.index.get_level_values(1) == model_n_train[1]
        )

        for feature in data.columns.get_level_values(0).unique():
            for metric in data.columns.get_level_values(1).unique():
                col = (feature, metric)
                is_min = data[col][group_mask] == data[col][group_mask].min()
                mask.loc[group_mask, col] = is_min

    return pd.DataFrame(np.where(mask, "font-weight: bold", ""), index=data.index, columns=data.columns)


def get_grouped_df(df: pd.DataFrame, multiple_effects: bool = False) -> pd.DataFrame:
    """
    Group the dataframe by the number of training samples, model, and split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the model training.
    multiple_effects : bool
        Whether the dataframe contains results for multiple effects (e.g., PD and ALE).

    Returns
    -------
    pd.DataFrame
        Grouped dataframe
    """
    if multiple_effects:
        pivoted = df.pivot_table(
            index=["effect", "sample_size", "model", "split"], columns=["feature", "metric"], values="value"
        ).reset_index()
        pivoted.columns.name = None

        return pivoted.groupby(by=["effect", "sample_size", "model", "split"]).mean()

    pivoted = df.pivot_table(
        index=["sample_size", "model", "split"], columns=["feature", "metric"], values="value"
    ).reset_index()

    pivoted.columns.name = None

    return pivoted.groupby(by=["sample_size", "model", "split"]).mean()
