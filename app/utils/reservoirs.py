import pandas as pd
import plotly.graph_objects as go


def get_busbar(config, node):
    for busbar in config.busbars:
        if busbar.name == node:
            return busbar
    return None


def get_gwh_max_volume(busbar):
    max_volume_gwh = 0.0
    for r in busbar.reservoirs():
        max_volume_gwh += r.metadata["reservoir_capacity_mm3"] * r.metadata["global_energy_equivalent"]

    return max_volume_gwh


def plot_reservoir(dff):
    fig = go.Figure()

    # --- Grey lines for each time series ---
    for iso_year in dff["iso_aar"].unique():
        df_year = dff[dff["iso_aar"] == iso_year]
        fig.add_trace(
            go.Scatter(
                x=df_year["iso_uke"],
                y=df_year["fyllingsgrad"] * 100,
                mode="lines",
                line=dict(color="grey", width=0.3),
                name=f"Year {iso_year}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --- Calculate values ---
    week_groups = dff.groupby("iso_uke")["fyllingsgrad"]
    min_values = week_groups.min()[:-1] * 100
    max_values = week_groups.max()[:-1] * 100
    mean_values = week_groups.mean()[:-1] * 100
    p25_values = week_groups.quantile(0.25)[:-1] * 100
    median_values = week_groups.quantile(0.5)[:-1] * 100
    p75_values = week_groups.quantile(0.75)[:-1] * 100

    fig.add_trace(
        go.Scatter(
            x=min_values.index,
            y=min_values.values,
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            name="Min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=max_values.index,
            y=max_values.values,
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            name="Max",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mean_values.index,
            y=mean_values.values,
            mode="lines",
            line=dict(color="blue", width=4),
            name="Mean",
        )
    )

    # --- Add percentiles as black lines ---
    fig.add_trace(
        go.Scatter(
            x=p25_values.index,
            y=p25_values.values,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="25th Percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=median_values.index,
            y=median_values.values,
            mode="lines",
            line=dict(color="green", width=2, dash="dash"),
            name="Median (50th)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p75_values.index,
            y=p75_values.values,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="75th Percentile",
        )
    )

    # --- Fill area between min and max ---
    fig.add_trace(
        go.Scatter(
            x=min_values.index,
            y=max_values.values,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),  # Invisible line for top bound
            showlegend=False,
            hoverinfo="skip",
            name="Max",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=min_values.index,
            y=min_values.values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(200,200,200,0.4)",  # grey fill with transparency
            line=dict(color="rgba(0,0,0,0)"),  # Invisible line for bottom bound
            showlegend=False,
            hoverinfo="skip",
            name="Min",
        )
    )

    area = dff["omrType"].iloc[0] + str(dff["omrnr"].iloc[0])
    # --- Layout ---
    fig.update_layout(
        title=f"Historical reservoir tracjectories for elsport area {area}",
        xaxis_title="Week",
        yaxis_title="Reservoir Filling [%]",
        yaxis=dict(range=[0, 100]),
        template="simple_white",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True)

    return fig


def plot_reservoir_together(
    dff_sim1: pd.DataFrame, dff_sim2: pd.DataFrame | None = None, sim1_name="Sim1", sim2_name="Sim2"
) -> go.Figure:
    fig = go.Figure()

    def add_percentile_traces(dff, label_prefix, color):
        week_groups = dff.groupby("iso_uke")["fyllingsgrad"]
        mean_values = week_groups.mean()[:-1] * 100
        p25 = week_groups.quantile(0.25)[:-1] * 100
        p50 = week_groups.quantile(0.5)[:-1] * 100
        p75 = week_groups.quantile(0.75)[:-1] * 100

        fig.add_trace(
            go.Scatter(
                x=mean_values.index,
                y=mean_values,
                mode="lines",
                name=f"{label_prefix} Mean",
                line=dict(color=color, width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p25.index,
                y=p25,
                mode="lines",
                name=f"{label_prefix} 25th %ile",
                line=dict(color=color, width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p50.index,
                y=p50,
                mode="lines",
                name=f"{label_prefix} Median",
                line=dict(color=color, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p75.index,
                y=p75,
                mode="lines",
                name=f"{label_prefix} 75th %ile",
                line=dict(color=color, width=1, dash="dot"),
            )
        )

    # Add first simulation
    add_percentile_traces(dff_sim1, sim1_name, "#8E0003")

    # Add second simulation if provided
    if dff_sim2 is not None:
        add_percentile_traces(dff_sim2, sim2_name, "#156082")

    area = dff_sim1["omrType"].iloc[0] + str(dff_sim1["omrnr"].iloc[0])
    fig.update_layout(
        title=f"Reservoir Percentiles – Elspot Area {area}",
        xaxis_title="Week",
        yaxis_title="Reservoir Filling [%]",
        yaxis=dict(range=[0, 100]),
        template="simple_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=True)

    return fig
