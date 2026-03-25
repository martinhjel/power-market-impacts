import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

def plot_grid_on_map(
    df_links: pd.DataFrame,
    df_nodes: pd.DataFrame,
    projection: str = "mercator",
    line_scale: float = 0.002,
) -> go.Figure:
    """
    Plot transmission links on a geographic map.

    Parameters
    ----------
    df_links
        dataframe with at least the columns ``node_from, node_to, capacity``.
    df_nodes
        CSV containing each node’s ``id, lat, lon``.
    projection
        Any Plotly `scattergeo` projection (e.g. ``"mercator"``, ``"orthographic"``).
    line_scale
        Multiplier converting *capacity* (MW) to line width (px).

    Returns
    -------
    plotly.graph_objects.Figure
        Fully configured figure ready for ``fig.show()``.
    """

    coord_map = df_nodes[["lat", "lon"]].apply(tuple, axis=1).to_dict()

    fig = go.Figure()

    # --- edges ---
    for _, row in df_links.iterrows():
        start = coord_map[row["node_from"]]
        end = coord_map[row["node_to"]]

        fig.add_trace(
            go.Scattergeo(
                lon=[start[1], end[1]],
                lat=[start[0], end[0]],
                mode="lines",
                line=dict(
                    width=max(row["capacity"] * line_scale, 1.0),
                    color="royalblue",
                ),
                hovertemplate=(
                    f"{row['node_from']} → {row['node_to']}<br>"
                    f"Capacity: {row['capacity']:,.0f} MW"
                ),
                showlegend=False,
            )
        )

        # Add midpoint label
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        fig.add_trace(
            go.Scattergeo(
                lon=[mid_lon],
                lat=[mid_lat],
                mode="text",
                text=[f"{int(row['capacity'])} MW"],
                textfont=dict(size=10, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --- lines with hover ---
    link_pairs = defaultdict(dict)

    for _, row in df_links.iterrows():
        pair = tuple(sorted([row["node_from"], row["node_to"]]))
        direction = (row["node_from"], row["node_to"])
        link_pairs[pair][direction] = row["capacity"]

    for (node_a, node_b), capacities in link_pairs.items():
        start = coord_map[node_a]
        end = coord_map[node_b]

        cap_ab = capacities.get((node_a, node_b), None)
        cap_ba = capacities.get((node_b, node_a), None)

        if cap_ab is not None and cap_ba is not None and cap_ab != cap_ba:
            cap_text = f"{int(cap_ab)} MW / {int(cap_ba)} MW"
            cap_val = max(cap_ab, cap_ba)
        else:
            cap_val = cap_ab or cap_ba
            cap_text = f"{int(cap_val)} MW"

        # Hoverable invisible marker at midpoint
        mid_lon = (start[1] + end[1]) / 2
        mid_lat = (start[0] + end[0]) / 2

        fig.add_trace(
            go.Scattergeo(
                lon=[mid_lon],
                lat=[mid_lat],
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                text=f"{node_a} → {node_b}<br>{cap_text}",
                hoverinfo="text",
                showlegend=False,
            )
        )

    # --- nodes ---
    fig.add_trace(
        go.Scattergeo(
            lon=df_nodes["lon"],
            lat=df_nodes["lat"],
            mode="markers+text",
            text=df_nodes.index,
            textposition="bottom center",
            marker=dict(size=6, color="black"),
            showlegend=False,
            hovertemplate="%{text}",
        )
    )

    fig.update_geos(
        projection_type=projection,
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        lataxis_range=[50, 72],  # Approximate latitude range for Northern Europe
        lonaxis_range=[-5, 35],  # Approximate longitude range
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title_text="Transmission Grid (capacity-weighted)",
        width=1200,
        height=800
    )
    return fig


def dummy():
    from collections import defaultdict

    # Group lines by unordered node pairs
    link_pairs = defaultdict(dict)

    for _, row in df_links.iterrows():
        pair = tuple(sorted([row["node_from"], row["node_to"]]))
        direction = (row["node_from"], row["node_to"])
        link_pairs[pair][direction] = row["capacity"]

    fig = go.Figure()

    # Plot each unique connection
    for (node_a, node_b), capacities in link_pairs.items():
        start = coord_map[node_a]
        end = coord_map[node_b]

        cap_ab = capacities.get((node_a, node_b), None)
        cap_ba = capacities.get((node_b, node_a), None)

        if cap_ab is not None and cap_ba is not None and cap_ab != cap_ba:
            cap_text = f"{int(cap_ab)} MW / {int(cap_ba)} MW"
            cap_val = max(cap_ab, cap_ba)
        else:
            cap_val = cap_ab or cap_ba
            cap_text = f"{int(cap_val)} MW"

        fig.add_trace(
            go.Scattergeo(
                lon=[start[1], end[1]],
                lat=[start[0], end[0]],
                mode="lines",
                line=dict(
                    width=max(cap_val * line_scale, 1.0),
                    color="royalblue",
                ),
                text=f"{node_a} → {node_b}<br>{cap_text}",
                hoverinfo="text",
                showlegend=False,
            )
        )
        
