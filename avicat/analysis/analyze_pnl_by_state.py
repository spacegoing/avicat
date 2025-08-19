import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_performance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates detailed performance statistics for a given DataFrame of P&L.
    Assumes the DataFrame is already grouped by state.
    """
    stats = {}
    for state, group in df.groupby("state"):
        if group["pl"].empty:
            continue

        total_pnl = group["pl"].sum()
        gross_profit = group["pl"][group["pl"] > 0].sum()
        gross_loss = abs(group["pl"][group["pl"] < 0].sum())

        stats[state] = {
            "trade_count": group["pl"].count(),
            "total_pnl": total_pnl,
            "average_pnl": group["pl"].mean(),
            "std_dev_pnl": group["pl"].std(),
            "win_rate_%": (group["pl"] > 0).sum() / group["pl"].count() * 100
            if group["pl"].count() > 0
            else 0,
            "profit_factor": gross_profit / gross_loss
            if gross_loss > 0
            else np.inf,
            "sharpe_ratio": (group["pl"].mean() / group["pl"].std())
            * np.sqrt(252)
            if group["pl"].std() > 0
            else 0,  # Annualized for daily-like data
        }

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.index.name = "state"
    return stats_df


def analyze_strategy_performance(
    states_file: Path,
    pnl_dir: Path,
    output_dir: Path,
):
    """
    Loads inferred states, finds all P&L files, merges them, and calculates performance.
    """
    print(f"📈 Loading inferred states from: {states_file}")
    try:
        states_df = pd.read_csv(states_file, parse_dates=["timestamp"])
        states_df.sort_values("timestamp", inplace=True)
    except FileNotFoundError:
        print(f"Error: States file not found at {states_file}")
        return

    if not pnl_dir.exists():
        print(f"Error: P&L directory not found at {pnl_dir}")
        return

    pnl_files = list(pnl_dir.glob("*.csv"))
    if not pnl_files:
        print(f"No P&L files (.csv) found in {pnl_dir}")
        return

    print(f"Found {len(pnl_files)} P&L files to analyze.")

    for pnl_file in pnl_files:
        print("\n" + "=" * 60)
        print(f"Processing strategy: {pnl_file.name}")
        print("=" * 60)

        # Load P&L data
        pnl_df = pd.read_csv(pnl_file)
        if "timestamp_uts" not in pnl_df.columns or "pl" not in pnl_df.columns:
            print(
                f"Warning: Skipping {pnl_file.name}. It must contain 'timestamp_uts' and 'pl' columns.",
            )
            continue

        pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp_uts"], unit="s")
        pnl_df.sort_values("timestamp", inplace=True)

        # Merge P&L data with states using the most recent state for each trade
        merged_df = pd.merge_asof(
            pnl_df,
            states_df[["timestamp", "state"]],
            on="timestamp",
            direction="backward",  # Finds the most recent state before or at the trade time
        )
        merged_df.dropna(subset=["state"], inplace=True)
        merged_df["state"] = merged_df["state"].astype(int)

        if merged_df.empty:
            print(
                "Warning: No matching states found for the timestamps in this P&L file. Check for timestamp alignment.",
            )
            continue

        # Calculate performance statistics
        performance_df = calculate_performance_stats(merged_df)

        # Display results
        print("📊 Performance by Market State:")
        print(performance_df.to_string(float_format="%.2f"))

        # Save results
        strategy_output_dir = output_dir / pnl_file.stem
        strategy_output_dir.mkdir(exist_ok=True, parents=True)

        output_path = strategy_output_dir / "performance_by_state.csv"
        performance_df.to_csv(output_path)
        print(f"\n💾 Saved detailed stats to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trading strategy P&L performance across different market states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--states-file",
        type=str,
        required=True,
        help="Path to the 'inferred_states_full.csv' file from a specific checkpoint analysis.",
    )
    parser.add_argument(
        "--pnl-dir",
        type=str,
        required=True,
        help="Directory containing the P&L CSV files for your strategies.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pnl_analysis_results",
        help="Directory to save the analysis results.",
    )
    args = parser.parse_args()

    analyze_strategy_performance(
        states_file=Path(args.states_file),
        pnl_dir=Path(args.pnl_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
