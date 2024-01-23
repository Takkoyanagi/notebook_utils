""" Modules for useful tooling in a notebook."""

import pandas as pd

def setup_pandas_config() -> None:
    """ removes truncation of dataframe
    tables that are viewed in the notebook.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("max_colwidth", None)

