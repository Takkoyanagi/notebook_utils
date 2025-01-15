""" Modules for ML model tooling in a notebook."""

import matplotlib.pyplot as plt
from typing import Optional
import pandas as pd
import seaborn as sns


def plot_feature_target_dependency(
    dataframe: pd.DataFrame,
    features: list,
    target: str,
    hue: Optional[str] = None,
    bins: int = 10,
    ignore_zero: bool = False,
    equisized_bins: bool = True,
):
    """Plots binned continuous features relative to the target."""
    df = dataframe.copy()
    for feature in features:
        if hue:
            df = dataframe.copy()[[feature, target, hue]]
        else:
            df = dataframe.copy()[[feature, target]]
        comment = ""
        plt.figure(layout="constrained")
        if df[feature].nunique() < bins:
            x = feature
        else:
            if ignore_zero:
                df = df[df[feature] != 0]
                comment = " - zeros have been removed"
            binning_cut = pd.qcut if equisized_bins else pd.cut
            df[feature + "_bin"] = binning_cut(df[feature], bins, duplicates="drop")
            x = feature + "_bin"
        ax1 = sns.pointplot(data=df, x=x, y=target, hue=hue)
        ax2 = ax1.twinx()
        sns.countplot(data=df, x=x, ax=ax2, alpha=0.25, hue=hue)
        ax1.set_title(feature + comment)
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
