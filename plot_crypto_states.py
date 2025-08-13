import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class CryptoStatePlotter:
    """Plot candlestick charts with state-colored backgrounds"""

    # State color palette (5 states by default, add more if needed)
    STATE_COLORS = {
        0: "rgba(220, 38, 127, 0.2)",  # Pink-red
        1: "rgba(255, 140, 0, 0.2)",  # Orange
        2: "rgba(255, 215, 0, 0.2)",  # Yellow
        3: "rgba(144, 238, 144, 0.2)",  # Light Green
        4: "rgba(34, 139, 34, 0.2)",  # Dark Green
        5: "rgba(147, 112, 219, 0.2)",  # Purple
        6: "rgba(70, 130, 180, 0.2)",  # Steel Blue
        7: "rgba(255, 20, 147, 0.2)",  # Deep Pink
    }

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_data(
        self,
        market_data: pd.DataFrame,
        states: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Prepare and align market data with states"""

        # Create state dataframe
        state_df = pd.DataFrame({"timestamp": timestamps, "state": states})

        # Ensure market_data has datetime index
        if "timestamp_utc" in market_data.columns:
            market_data["timestamp"] = pd.to_datetime(
                market_data["timestamp_utc"],
            )
        elif "timestamp" in market_data.columns:
            market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
        else:
            raise ValueError("No timestamp column found in market data")

        # Merge market data with states
        merged_df = pd.merge(
            market_data,
            state_df,
            on="timestamp",
            how="inner",
        )

        # Sort by timestamp
        merged_df = merged_df.sort_values("timestamp")

        return merged_df

    def plot_month(
        self,
        df_month: pd.DataFrame,
        year: int,
        month: int,
        save_path: Optional[Path] = None,
    ) -> go.Figure:
        """Create candlestick plot with volume for one month"""

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"BTC Price - {year:04d}-{month:02d}", "Volume"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Add candlestick chart first
        fig.add_trace(
            go.Candlestick(
                x=df_month["timestamp"],
                open=df_month["btc_open"],
                high=df_month["btc_high"],
                low=df_month["btc_low"],
                close=df_month["btc_close"],
                name="BTC",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df_month["timestamp"],
                y=df_month["btc_volume"],
                name="Volume",
                marker_color="rgba(100, 100, 100, 0.5)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Get y-axis ranges after adding data
        fig.update_layout(showlegend=True)

        # Calculate y-axis ranges for both subplots
        price_min = df_month[["btc_low"]].min().min() * 0.99
        price_max = df_month[["btc_high"]].max().max() * 1.01
        volume_max = df_month["btc_volume"].max() * 1.1

        # Prepare all background shapes
        shapes = []

        # Calculate bar width (for hourly data)
        timestamps = df_month["timestamp"].values
        if len(timestamps) > 1:
            # Calculate median time difference
            time_diffs = pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1])
            bar_width = time_diffs.median()
        else:
            bar_width = pd.Timedelta(hours=1)

        half_width = bar_width / 2

        for i in range(len(df_month)):
            timestamp = pd.Timestamp(timestamps[i])
            state = df_month.iloc[i]["state"]
            color = self.STATE_COLORS.get(state, "rgba(128, 128, 128, 0.2)")

            # Shape for price chart (row 1) - explicitly use x and y for first subplot
            shapes.append(
                dict(
                    type="rect",
                    x0=timestamp - half_width,
                    x1=timestamp + half_width,
                    y0=price_min,
                    y1=price_max,
                    xref="x",
                    yref="y",
                    fillcolor=color,
                    layer="below",
                    line=dict(width=0),  # Changed from line_width=0
                ),
            )

            # Shape for volume chart (row 2)
            shapes.append(
                dict(
                    type="rect",
                    x0=timestamp - half_width,
                    x1=timestamp + half_width,
                    y0=0,
                    y1=volume_max,
                    xref="x2",
                    yref="y2",
                    fillcolor=color,
                    layer="below",
                    line=dict(width=0),  # Changed from line_width=0
                ),
            )

        # Add all shapes at once
        fig.update_layout(shapes=shapes)

        # Update layout
        fig.update_layout(
            title=f"BTC Market Regimes - {year:04d}-{month:02d}",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
            spikedistance=-1,
        )

        # Update x-axes to show spike lines
        fig.update_xaxes(
            showspikes=True,
            spikecolor="grey",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=0.5,
        )

        # Update y-axes with proper ranges
        fig.update_yaxes(
            title_text="Price (USD)",
            row=1,
            col=1,
            range=[price_min, price_max],
            showspikes=True,
        )
        fig.update_yaxes(
            title_text="Volume",
            row=2,
            col=1,
            range=[0, volume_max],
            showspikes=True,
        )

        # Format x-axis dates
        fig.update_xaxes(
            title_text="Date",
            row=2,
            col=1,
            tickformat="%Y-%m-%d %H:%M",
            tickmode="auto",
            nticks=5,
            type="date",
        )

        # Add legend for states
        self.add_state_legend(fig, df_month["state"].unique())

        if save_path:
            # Force update before saving PNG to ensure shapes are rendered
            fig.update_layout(
                shapes=shapes,
                autosize=False,
                width=1400,
                height=800,
            )

            # Update traces to ensure they're on correct axes
            fig.update_traces(xaxis="x", yaxis="y", row=1, col=1)
            fig.update_traces(xaxis="x2", yaxis="y2", row=2, col=1)

            # # Save as static image with explicit backend
            # fig.write_image(
            #     str(save_path),
            #     width=1400,
            #     height=800,
            #     engine="kaleido",
            #     format="png",
            #     scale=2,  # Higher resolution
            # )

            # # Also save as interactive HTML
            fig.write_html(str(save_path).replace(".png", ".html"))
            print(f"Saved plot to {save_path}")

        return fig

    def add_state_legend(self, fig: go.Figure, states: np.ndarray):
        """Add legend for state colors - using only numbers"""

        # Add invisible scatter traces for legend
        for state in sorted(states):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=f"State {state}",
                    marker=dict(
                        size=10,
                        color=self.STATE_COLORS.get(
                            state,
                            "rgba(128, 128, 128, 0.5)",
                        ),
                    ),
                    showlegend=True,
                ),
            )

    def plot_all_months(
        self,
        df: pd.DataFrame,
        output_prefix: str = "btc_states",
    ) -> List[Path]:
        """Plot all months in the dataset"""

        # Extract year and month
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # Get unique year-month combinations
        year_months = df.groupby(["year", "month"]).size().index

        saved_paths = []

        for year, month in year_months:
            # Filter data for this month
            df_month = df[(df["year"] == year) & (df["month"] == month)].copy()

            if len(df_month) < 2:  # Skip if too few data points
                continue

            # Create save path
            save_path = (
                self.output_dir / f"{output_prefix}_{year:04d}_{month:02d}.png"
            )

            # Create and save plot
            self.plot_month(df_month, year, month, save_path)
            saved_paths.append(save_path)

        return saved_paths

    def create_summary_plot(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> go.Figure:
        """Create a summary plot showing state distribution over time"""

        # Resample to daily for summary
        df_daily = (
            df.set_index("timestamp")
            .resample("D")
            .agg(
                {
                    "state": lambda x: x.mode()[0] if len(x) > 0 else np.nan,
                    "btc_close": "last",
                    "btc_volume": "sum",
                },
            )
            .dropna()
        )

        # Create figure
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("BTC Price", "Market Regime", "Volume"),
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ],
        )

        # Add traces first
        fig.add_trace(
            go.Scatter(
                x=df_daily.index,
                y=df_daily["btc_close"],
                mode="lines",
                name="BTC Price",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_daily.index,
                y=df_daily["state"],
                mode="lines",
                name="State",
                line=dict(color="black", width=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_daily.index,
                y=df_daily["btc_volume"],
                name="Volume",
                marker_color="rgba(100, 100, 100, 0.5)",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Calculate y-ranges
        price_min = df_daily["btc_close"].min() * 0.99
        price_max = df_daily["btc_close"].max() * 1.01
        state_max = df_daily["state"].max() + 0.5
        volume_max = df_daily["btc_volume"].max() * 1.1

        # Create all background shapes
        shapes = []
        daily_index = df_daily.index.values
        daily_states = df_daily["state"].values

        for i in range(len(df_daily) - 1):
            state = int(daily_states[i])
            x_start = pd.Timestamp(daily_index[i])
            x_end = pd.Timestamp(daily_index[i + 1])
            color = self.STATE_COLORS.get(state, "rgba(128, 128, 128, 0.2)")

            # Shape for state plot (row 2)
            shapes.append(
                dict(
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=0,
                    y1=state_max,
                    xref="x2",
                    yref="y2",
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                ),
            )

        # Add all shapes at once
        fig.update_layout(shapes=shapes)

        # Update layout
        fig.update_layout(
            title="BTC Market Regimes - Full Period Summary",
            height=900,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        # Update axes
        fig.update_yaxes(
            title_text="Price (USD)",
            row=1,
            col=1,
            range=[price_min, price_max],
        )
        fig.update_yaxes(
            title_text="State",
            row=2,
            col=1,
            range=[0, state_max],
        )
        fig.update_yaxes(
            title_text="Volume",
            row=3,
            col=1,
            range=[0, volume_max],
        )

        # Format x-axis
        fig.update_xaxes(
            title_text="Date",
            row=3,
            col=1,
            tickformat="%Y-%m-%d",
            type="date",
        )

        # Add state legend
        unique_states = df_daily["state"].dropna().unique()
        self.add_state_legend(fig, unique_states)

        if save_path:
            # fig.write_image(
            #     str(save_path),
            #     width=1400,
            #     height=900,
            #     engine="kaleido",
            # )
            fig.write_html(str(save_path).replace(".png", ".html"))
            print(f"Saved summary plot to {save_path}")

        return fig


if __name__ == "__main__":
    # Test with dummy data
    plotter = CryptoStatePlotter()

    # Create dummy data
    dates = pd.date_range("2024-01-01", "2024-01-07", freq="H")
    n = len(dates)

    price_base = 40000
    dummy_df = pd.DataFrame(
        {
            "timestamp": dates,
            "btc_open": price_base + np.random.randn(n).cumsum() * 100,
            "btc_high": price_base + np.random.randn(n).cumsum() * 100 + 50,
            "btc_low": price_base + np.random.randn(n).cumsum() * 100 - 50,
            "btc_close": price_base + np.random.randn(n).cumsum() * 100,
            "btc_volume": np.random.exponential(1000, n),
            "state": np.random.randint(0, 5, n),
        },
    )

    # Test plotting
    paths = plotter.plot_all_months(dummy_df, "test_plot")
    print(f"Created {len(paths)} plots")
