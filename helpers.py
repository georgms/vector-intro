from typing import List
from urllib.parse import quote_plus, urlunparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from annoy import AnnoyIndex
from IPython.core.display import HTML
from IPython.display import display
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA


def get_torch_device_name() -> str:
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    return device_name


def _display_url_images(image_urls, min_width=150, max_columns=6):
    columns = max(1, min(max_columns, len(image_urls)))
    # Generate HTML for the flexible grid
    html = f'<div style="display: flex; flex-wrap: wrap;">'
    for url in image_urls:
        html += f'<div style="flex: 1; min-width: {min_width}px; max-width: {100 / columns:.2f}; padding: 10px;">'
        html += (
            f'<img src="{url}" style="width: 100%; height: auto;">'
        )
        html += f"</div>"
    html += "</div>"
    # Display the HTML
    display(HTML(html))


def display_product_images(
    product_ids: List[str],
    merchant: str,
    min_width=150,
    max_columns=4,
):
    image_urls = [
        get_thumb_image_url(merchant, p) for p in product_ids
    ]
    _display_url_images(
        image_urls, min_width=min_width, max_columns=max_columns
    )


def get_thumb_image_url(merchant_id: str, product_id: str) -> str:
    scheme = "https"
    netloc = "thumbs.nosto.com"
    path = (
        f"quick/{quote_plus(merchant_id)}/8/{quote_plus(product_id)}"
    )
    return urlunparse((scheme, netloc, path, "", "", ""))


def display_images_and_names(df, merchant_id, header_text=None):
    html = "<table style='width:100%'>"

    if header_text:
        html += (
            f"<tr>"
            f"<th colspan='{len(df)}' "
            f"style='text-align:left'>"
            f"<h2 style='max-width:100%;"
            f"overflow-wrap: break-word'>{header_text}</h2>"
            f"</th></tr>"
        )

    html += "<tr>"

    for index, row in df.iterrows():
        image_url = get_thumb_image_url(merchant_id, row["productId"])
        html += (
            f"<td style='width:25%; text-align:center'>"
            f"<img src='{image_url}' style='max-width:100%' title='{row['name']}'>"
            f"<br>{row['name']}"
            f"</td>"
        )
    html += "</tr></table>"
    return html


def get_html_image(merchant, p, width=50):
    link = get_thumb_image_url(merchant, p)
    return (
        f'<img src="{link}" style="width: {width}%; height: auto;">'
    )


def color_embedings_df(
    df: pd.DataFrame,
    color_col: str = None,
    hover_data: list = None,
    dimensions: int = 2,
    add_vectors: bool = False,
):
    embeddings_pca = _get_transform_embedding_df(
        color_col, df, dimensions, hover_data
    )
    scatter_kwargs = dict(
        x="x",
        y="y",
        z="z",
        color=color_col,
        hover_data=hover_data,
    )

    scatter = px.scatter_3d
    if dimensions == 2:
        scatter_kwargs.pop("z")
        scatter = px.scatter
    fig = scatter(embeddings_pca, **scatter_kwargs)
    if add_vectors:
        fig = _plot_vectors(fig, embeddings_pca, dimensions)
    helper = embeddings_pca.loc[:, ["x", "y"]]
    min_helper = helper.apply(min).min() - 0.01
    max_helper = helper.apply(max).max() + 0.01
    layout = {
        f"{x}axis": {"range": [min_helper, max_helper]}
        for x in helper.columns
    }

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_layout(
        title=f"Word Embeddings {dimensions}D Visualization with Color",
        autosize=False,
        **layout,
    )
    return fig


def _get_transform_embedding_df(
    color_col, df, dimensions, hover_data
):
    if dimensions not in [2, 3]:
        raise ValueError("'dimensions' should be either 2 or 3")
    embed_array = df["embeddings"].to_list()
    check = len(embed_array[0]) != dimensions
    if check:
        pca = PCA(n_components=dimensions)
        embed_array = pca.fit_transform(embed_array)
    embeddings_pca = pd.DataFrame(
        embed_array, columns=list("xyz")[:dimensions]
    )
    if color_col is not None:
        embeddings_pca[color_col] = df.loc[:, color_col].values
    if hover_data is not None:
        for col in hover_data:
            embeddings_pca[col] = df.loc[:, col].values
    return embeddings_pca


def _plot_vectors(fig, embeddings_pca, dimensions):
    scatters = []
    Scatter = go.Scatter3d if dimensions == 3 else go.Scatter

    for i, row in embeddings_pca.iterrows():
        # Create a list with dimension number of zeroes

        coordinates = {
            dimension: [0, row[dimension]]
            for dimension in ["x", "y", "z"][:dimensions]
        }

        # Add dashed arrow from origin to the point
        scatters.append(
            Scatter(
                **{
                    **coordinates,
                    **dict(
                        mode="lines",
                        line=dict(
                            color="black", width=0.5, dash="dash"
                        ),
                        marker=dict(size=0),
                        showlegend=False,
                    ),
                }
            )
        )
    fig.add_traces(scatters)

    return fig


def calc_distance_matrix(index_ann: AnnoyIndex) -> pd.DataFrame:
    num_items = index_ann.get_n_items()
    num_neighbors = int(num_items / index_ann.get_n_trees() + 0.5)
    distance_df = pd.DataFrame(
        np.nan,
        index=np.arange(num_items),
        columns=np.arange(num_items),
    )

    for col in distance_df.columns:
        neighbors, distances = index_ann.get_nns_by_item(
            col, n=num_neighbors, include_distances=True
        )
        distance_df.loc[neighbors, col] = distances

    return distance_df


def calc_cluster(index_ann: AnnoyIndex) -> pd.Series:
    """
    Calculate kmeans clusters using the distances in an Annoy Index

    :param AnnoyIndex index_ann: the annoy index
    :return: a series with the cluster each item in the annoy index belongs to
    :rtype: pd.Series
    """

    def fill_na_with_max(df: pd.DataFrame):
        """
        Replaces NaN values in a DataFrame with the max value in the DataFrame multiplied by its shape

        :param pd.DataFrame df: the DataFrame to replace NaNs in
        :return: DataFrame which NaNs replaced
        :rtype: pd.DataFrame
        """
        max_value = df.max().max() * df.shape[0]
        df.fillna(max_value, inplace=True)
        return df

    distance_df = calc_distance_matrix(index_ann)
    distance_df = fill_na_with_max(distance_df)

    kmeans = KMeans(
        n_clusters=int(index_ann.get_n_trees()),
        random_state=42,
        n_init="auto",
        max_iter=index_ann.get_n_items(),
    )
    kmeans.fit(distance_df)
    cluster = kmeans.predict(distance_df)

    return pd.Series(cluster).astype("category")
