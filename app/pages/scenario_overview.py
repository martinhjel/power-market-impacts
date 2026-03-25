"""
Scenario Overview Page

Plots mean power price, hydropower production, load for each area across all nodes,
as well as transmission utilization for one scenario using matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lpr_sintef_bifrost.ltm import LTM
from lpr_sintef_bifrost.utils.dataframe import df_from_pyltm_result

st.set_page_config(layout="wide")
st.title("📊 Scenario Overview")

# Select simulation folder
result_paths = sorted(
    [i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()],
    key=lambda p: p.name.lower(),
)


def _result_label(p: Path) -> str:
    """Return a short label for a result path."""
    parent = p.parent.name
    if "1H" in parent:
        res = "1H"
    elif "1D" in parent:
        res = "1D"
    else:
        res = parent
    return f"{p.name} ({res})"


selected_path = st.sidebar.selectbox(
    "Select simulation:",
    result_paths,
    format_func=_result_label,
    index=0,
)

st.sidebar.markdown(f"**Path:** `{selected_path.name}`")


@st.cache_resource
def load_model(path: Path):
    """Load the LTM model from the scenario folder."""
    session = LTM.session_from_folder(path / "run_folder/emps")
    return session.model


# Load model
with st.spinner("Loading model..."):
    model = load_model(selected_path)

busbars = {b.name: b for b in model.busbars()}
busbar_names = sorted(busbars.keys())

st.info(f"Loaded {len(busbars)} busbars: {', '.join(busbar_names[:10])}{'...' if len(busbars) > 10 else ''}")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💶 Prices", "🚰 Hydro Production", "🔌 Load", "🔄 Transmission", "📈 Time Series"]
)

with tab1:
    st.header("Average Power Prices by Area")

    with st.spinner("Computing prices..."):
        price_data = {}
        for name, busbar in busbars.items():
            try:
                df_price = df_from_pyltm_result(busbar.market_result_price())
                # Mean across scenarios, then mean across time
                price_data[name] = df_price.mean().mean()
            except Exception as e:
                st.warning(f"Could not load price for {name}: {e}")
                price_data[name] = np.nan

        df_prices = pd.Series(price_data).dropna().sort_values(ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_prices) * 0.3)))
        bars = ax.barh(df_prices.index, df_prices.values, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Average Price (EUR/MWh)", fontsize=12)
        ax.set_ylabel("Area", fontsize=12)
        ax.set_title("Average Power Price by Area", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(df_prices.items()):
            ax.text(val, i, f" {val:.1f}", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Statistics")
        st.write(f"**Mean:** {df_prices.mean():.2f} EUR/MWh")
        st.write(f"**Median:** {df_prices.median():.2f} EUR/MWh")
        st.write(f"**Std Dev:** {df_prices.std():.2f} EUR/MWh")
        st.write(f"**Min:** {df_prices.min():.2f} EUR/MWh ({df_prices.idxmin()})")
        st.write(f"**Max:** {df_prices.max():.2f} EUR/MWh ({df_prices.idxmax()})")

        st.subheader("Data Table")
        st.dataframe(df_prices.to_frame(name="Price (EUR/MWh)"), height=400)

with tab2:
    st.header("Total Hydropower Production by Area")

    with st.spinner("Computing hydro production..."):
        hydro_data = {}
        for name, busbar in busbars.items():
            try:
                df_hydro = df_from_pyltm_result(busbar.sum_hydro_production())
                # Mean across scenarios, then sum across time (total energy)
                hydro_data[name] = df_hydro.mean(axis=1).sum()
            except Exception:
                # Some areas may not have hydro
                hydro_data[name] = 0

        df_hydro = pd.Series(hydro_data).sort_values(ascending=False)
        df_hydro = df_hydro[df_hydro > 0]  # Remove zeros

    if len(df_hydro) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar plot
            fig, ax = plt.subplots(figsize=(12, max(6, len(df_hydro) * 0.3)))
            bars = ax.barh(df_hydro.index, df_hydro.values, color="dodgerblue", edgecolor="black", linewidth=0.5)
            ax.set_xlabel("Total Production (MWh)", fontsize=12)
            ax.set_ylabel("Area", fontsize=12)
            ax.set_title("Total Hydropower Production by Area", fontsize=14, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, (idx, val) in enumerate(df_hydro.items()):
                ax.text(val, i, f" {val / 1e6:.1f} TWh", va="center", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Statistics")
            st.write(f"**Total:** {df_hydro.sum() / 1e6:.2f} TWh")
            st.write(f"**Mean:** {df_hydro.mean() / 1e6:.2f} TWh")
            st.write(f"**Median:** {df_hydro.median() / 1e6:.2f} TWh")
            st.write(f"**Max:** {df_hydro.max() / 1e6:.2f} TWh ({df_hydro.idxmax()})")

            st.subheader("Data Table")
            df_hydro_display = (df_hydro / 1e6).to_frame(name="Production (TWh)")
            st.dataframe(df_hydro_display, height=400)
    else:
        st.warning("No hydropower production data found.")

with tab3:
    st.header("Total Load by Area")

    with st.spinner("Computing load..."):
        load_data = {}
        for name, busbar in busbars.items():
            try:
                df_load = df_from_pyltm_result(busbar.sum_load())
                # Mean across scenarios, then sum across time (total energy consumed)
                load_data[name] = df_load.mean(axis=1).sum()
            except Exception as e:
                st.warning(f"Could not load data for {name}: {e}")
                load_data[name] = 0

        df_load = pd.Series(load_data).sort_values(ascending=False)
        df_load = df_load[df_load > 0]  # Remove zeros

    if len(df_load) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar plot
            fig, ax = plt.subplots(figsize=(12, max(6, len(df_load) * 0.3)))
            bars = ax.barh(df_load.index, df_load.values, color="coral", edgecolor="black", linewidth=0.5)
            ax.set_xlabel("Total Load (MWh)", fontsize=12)
            ax.set_ylabel("Area", fontsize=12)
            ax.set_title("Total Load by Area", fontsize=14, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, (idx, val) in enumerate(df_load.items()):
                ax.text(val, i, f" {val / 1e6:.1f} TWh", va="center", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Statistics")
            st.write(f"**Total:** {df_load.sum() / 1e6:.2f} TWh")
            st.write(f"**Mean:** {df_load.mean() / 1e6:.2f} TWh")
            st.write(f"**Median:** {df_load.median() / 1e6:.2f} TWh")
            st.write(f"**Max:** {df_load.max() / 1e6:.2f} TWh ({df_load.idxmax()})")

            st.subheader("Data Table")
            df_load_display = (df_load / 1e6).to_frame(name="Load (TWh)")
            st.dataframe(df_load_display, height=400)
    else:
        st.warning("No load data found.")

with tab4:
    st.header("Transmission Line Utilization")

    with st.spinner("Computing transmission flows..."):
        transmission_data = []
        for line in model.dclines():
            try:
                _, data = line.transmission_results()
                flow_array = np.array(data)
                # Mean across scenarios and time
                mean_flow = flow_array.mean()
                max_flow = flow_array.max()
                min_flow = flow_array.min()

                transmission_data.append(
                    {
                        "line": line.name,
                        "mean_flow": mean_flow,
                        "max_flow": max_flow,
                        "min_flow": min_flow,
                        "abs_mean": abs(mean_flow),
                    }
                )
            except Exception as e:
                st.warning(f"Could not load transmission data for {line.name}: {e}")

        df_transmission = pd.DataFrame(transmission_data)

        if len(df_transmission) > 0:
            df_transmission = df_transmission.sort_values("abs_mean", ascending=False)

    if len(df_transmission) > 0:
        # Filter options
        show_top_n = st.slider("Show top N lines:", 10, len(df_transmission), min(20, len(df_transmission)))
        df_top = df_transmission.head(show_top_n)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar plot with positive/negative flows
            fig, ax = plt.subplots(figsize=(12, max(6, len(df_top) * 0.3)))

            colors = ["green" if x >= 0 else "red" for x in df_top["mean_flow"]]
            bars = ax.barh(
                df_top["line"], df_top["mean_flow"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.7
            )

            ax.set_xlabel("Average Flow (MW)", fontsize=12)
            ax.set_ylabel("Transmission Line", fontsize=12)
            ax.set_title(f"Top {show_top_n} Transmission Lines by Average Flow", fontsize=14, fontweight="bold")
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, (idx, row) in enumerate(df_top.iterrows()):
                val = row["mean_flow"]
                ax.text(val, i, f" {val:.0f} MW", va="center", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # Additional plot: Min/Max range
            st.subheader("Flow Range (Min/Max)")
            fig2, ax2 = plt.subplots(figsize=(12, max(6, len(df_top) * 0.3)))

            y_pos = np.arange(len(df_top))
            ax2.barh(
                y_pos,
                df_top["max_flow"] - df_top["min_flow"],
                left=df_top["min_flow"],
                color="lightblue",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            ax2.scatter(df_top["mean_flow"], y_pos, color="red", s=50, zorder=3, label="Mean")

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(df_top["line"])
            ax2.set_xlabel("Flow (MW)", fontsize=12)
            ax2.set_ylabel("Transmission Line", fontsize=12)
            ax2.set_title("Flow Range (Min to Max) with Mean", fontsize=14, fontweight="bold")
            ax2.axvline(x=0, color="black", linewidth=0.8)
            ax2.grid(axis="x", alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig2)

        with col2:
            st.subheader("Statistics")
            st.write(f"**Total lines:** {len(df_transmission)}")
            st.write(f"**Avg absolute flow:** {df_transmission['abs_mean'].mean():.2f} MW")
            st.write(f"**Max flow:** {df_transmission['max_flow'].max():.2f} MW")
            st.write(f"**Min flow:** {df_transmission['min_flow'].min():.2f} MW")

            st.subheader(f"Top {show_top_n} Lines")
            df_display = df_top[["line", "mean_flow", "min_flow", "max_flow"]].copy()
            df_display.columns = ["Line", "Mean (MW)", "Min (MW)", "Max (MW)"]
            st.dataframe(
                df_display.style.format(
                    {
                        "Mean (MW)": "{:.1f}",
                        "Min (MW)": "{:.1f}",
                        "Max (MW)": "{:.1f}",
                    }
                ),
                height=400,
                hide_index=True,
            )
    else:
        st.warning("No transmission data found.")

with tab5:
    st.header("Time Series Analysis")
    st.write("Mean values over scenarios, plotted over simulation time")

    # Select areas to plot
    selected_areas = st.multiselect(
        "Select areas to plot:",
        busbar_names,
        default=[name for name in ["NO1", "NO2", "NO3", "NO4", "NO5"] if name in busbar_names][:5],
    )

    if not selected_areas:
        st.warning("Please select at least one area.")
    else:
        # Metric selection
        metric = st.radio("Select metric:", ["Price", "Hydro Production", "Load", "Reservoir"], horizontal=True)

        with st.spinner(f"Loading {metric.lower()} time series..."):
            fig, ax = plt.subplots(figsize=(14, 7))

            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_areas)))

            for i, area in enumerate(selected_areas):
                try:
                    busbar = busbars[area]

                    if metric == "Price":
                        df = df_from_pyltm_result(busbar.market_result_price())
                        ylabel = "Price (EUR/MWh)"
                    elif metric == "Hydro Production":
                        df = df_from_pyltm_result(busbar.sum_hydro_production())
                        ylabel = "Hydro Production (MW)"
                    elif metric == "Load":
                        df = df_from_pyltm_result(busbar.sum_load())
                        ylabel = "Load (MW)"
                    else:  # Reservoir
                        df = df_from_pyltm_result(busbar.sum_reservoir())
                        ylabel = "Reservoir Level (GWh)"

                    # Mean across scenarios
                    mean_series = df.mean(axis=1)

                    ax.plot(mean_series.index, mean_series.values, label=area, linewidth=2, color=colors[i], alpha=0.8)

                except Exception as e:
                    st.warning(f"Could not load {metric.lower()} for {area}: {e}")

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{metric} Over Time (Mean Across Scenarios)", fontsize=14, fontweight="bold")
            ax.legend(loc="best", fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Optional: Add aggregated statistics
            if st.checkbox("Show aggregated statistics"):
                st.subheader("Time-Averaged Statistics by Area")

                stats_data = {}
                for area in selected_areas:
                    try:
                        busbar = busbars[area]

                        if metric == "Price":
                            df = df_from_pyltm_result(busbar.market_result_price())
                        elif metric == "Hydro Production":
                            df = df_from_pyltm_result(busbar.sum_hydro_production())
                        elif metric == "Load":
                            df = df_from_pyltm_result(busbar.sum_load())
                        else:  # Reservoir
                            df = df_from_pyltm_result(busbar.sum_reservoir())

                        mean_series = df.mean(axis=1)
                        stats_data[area] = {
                            "Mean": mean_series.mean(),
                            "Std": mean_series.std(),
                            "Min": mean_series.min(),
                            "Max": mean_series.max(),
                        }
                    except:
                        pass

                df_stats = pd.DataFrame(stats_data).T
                st.dataframe(df_stats.style.format("{:.2f}"), height=300)

# Footer
st.divider()
st.caption(f"Scenario: {selected_path.name}")
