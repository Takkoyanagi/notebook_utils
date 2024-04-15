""" Modules for eda tooling in a notebook."""

import datetime
from typing import Optional, Any

from bokeh.io import output_notebook, show, save, output_file
from bokeh.models import Range1d, ColumnDataSource, ColorBar, LinearInterpolator, Interpolator
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.plotting import figure
from bokeh.palettes import Spectral6, Category20c, Category10, Category20b
from bokeh.transform import linear_cmap, factor_cmap
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from bokeh.embed import file_html
from bokeh.resources import CDN
import pandas as pd


def setup_pandas_config() -> None:
    """ removes truncation of dataframe
    tables that are viewed in the notebook.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("max_colwidth", None)


def project_data_umap(n_neighbors: int, metric: str, df: pd.DataFrame, umap_exclude_cols:list, embedded_col:str=None)->np.ndarray:
    """ Project high dimensional data down to two dimensions
    using UMAP.
    """
    assert df.isna().sum(axis=0).sum() == 0, 'DataFrame can not have null values'
    if not umap_exclude_cols:
        umap_exclude_cols = []
    _umap = UMAP(
    n_neighbors=n_neighbors,
    n_components=2,
    metric=metric,
    n_jobs=1,
    random_state=42,
    )
    if not embedded_col:
        scaler = StandardScaler()
        numerical_cols = df.describe().columns
        numerical_cols = [col for col in numerical_cols if col not in umap_exclude_cols]
        data = scaler.fit_transform(df[numerical_cols])
    else:
        data = np.array([*df[embedded_col].to_numpy()])
    return _umap.fit_transform(data)


def plot_bokeh(
    df: pd.DataFrame,
    n_neighbors: int,
    metric: str,
    title: str,
    hover_fields: list,
    umap_data: Optional[np.array]=None,
    embedded_col: Optional[str]=None,
    fill_color_field: Optional[str]=None,
    fill_color_rev:Optional[bool]=None,
    outline_color_field: Optional[str]=None,
    outline_color_rev: Optional[bool]=None,
    size_field: Optional[str] = None,
    umap_exclude_cols: list = None,
    save_destination: bool = False,
    tab: bool = False,
    databricks: bool=False,
    output_html: bool=False,
) -> Optional[figure]:
    """Creates an interactive Bokeh scatter plot
    based on the parameters passed."""

    if umap_data is None:
        umap_emb = project_data_umap(n_neighbors=n_neighbors,
                                    metric=metric,
                                    df=df,
                                    umap_exclude_cols=umap_exclude_cols,
                                    embedded_col=embedded_col)
    else:
        umap_emb = umap_data
    viz_df = df.copy()
    viz_df["umap_x"] = umap_emb[:, 0]
    viz_df["umap_y"] = umap_emb[:, 1]

    # Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [(col, "@" + col) for col in hover_fields]

    # Create a plot — set dimensions, toolbar, and title
    plot = figure(
        tooltips=HOVER_TOOLTIPS,
        tools="pan,wheel_zoom,save,reset",
        active_scroll="wheel_zoom",
        x_range=Range1d(-10.1, 10.1),
        y_range=Range1d(-10.1, 10.1),
        title=title,
        width=1000,
        height=750,
    )
    mapper_fill = 'blue'
    if fill_color_field:
        if isinstance(df[fill_color_field].iloc[0], str):
            colors = list(Category20c[20]) + list(Category20b[20]) + list(Category10[10])
            mapper_fill = factor_cmap(field_name=fill_color_field, palette=colors, factors=list(df[fill_color_field].unique()),
                                      start=1,
                                      end=len(df[fill_color_field].unique())+1)
        else:
            fill_palette = Spectral6[::-1] if fill_color_rev else Spectral6
            mapper_fill = linear_cmap(
                field_name=fill_color_field,
                palette=fill_palette,
                low=min(viz_df[fill_color_field]),
                high=max(viz_df[fill_color_field]),
            )

        color_bar_fill = ColorBar(
            color_mapper=mapper_fill["transform"],
            width=8,
            location=(0, 0),
            title=fill_color_field.replace("_", " ").title(),
        )

    mapper_outline = None
    if outline_color_field:
        outline_palette = Spectral6[::-1] if outline_color_rev else Spectral6
        mapper_outline = linear_cmap(
            field_name=outline_color_field, palette=outline_palette, low=0.0, high=1.0
        )
        color_bar_outline = ColorBar(
            color_mapper=mapper_outline["transform"],
            width=8,
            location=(0, 0),
            title=outline_color_field.replace("_", " ").title(),
        )

    if size_field:
        viz_df[f'scaled_{size_field}'] = [math.sqrt(val) for val in viz_df[size_field]]
        size_interpolator_range = [viz_df[f'scaled_{size_field}'].min(), viz_df[f'scaled_{size_field}'].max()]
        size_mapper = LinearInterpolator(x=size_interpolator_range, y=size_interpolator_range)
        size = {"field": f'scaled_{size_field}', "transform": size_mapper}
    else:
        size = 10
    plot.scatter(
        x="umap_x",
        y="umap_y",
        source=viz_df,
        fill_alpha=0.6,
        line_color=mapper_outline,
        color=mapper_fill,
        size=size,
        marker="circle_dot",
    )
    if fill_color_field:
        plot.add_layout(color_bar_fill, "right")
    if outline_color_field:
        plot.add_layout(color_bar_outline, "right")

    if databricks:
        # create an html document that embeds the Bokeh plot
        html = file_html(plot, CDN, title)

        # display this html
        if save_destination:
            output_file('./cluster_plot.html')
        else:
            displayHTML(html)

    if tab:
        return plot
    else:
        if save_destination:
            today = datetime.datetime.today().date()
            figname = f"{today}_{title}_n_neigh_{n_neighbors}_{metric}"
            save(
                plot,
                filename=figname + ".html",
            )
        show(plot)


def plot_bokeh_tabs(bokeh_plots: dict) -> None:
    """WIP usees tabs."""
    )
    tabs = []
    for title, plot in bokeh_plots.items():
        tabs.append(TabPanel(child=plot, title)
    show(Tabs(tabs=*tabs))
