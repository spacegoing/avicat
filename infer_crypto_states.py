import json
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


from model import DeepStateSpaceModel
from plot_crypto_states import CryptoStatePlotter


class CryptoStateInference:
    """Infer market states from trained DSSM model"""

    def __init__(
        self,
        checkpoint_path: str,
        market_data_path: str = "market_data/merged_market_data_20250216.csv",
        regime_data_path: str = "market_data/market_regime_features_20250216.csv",
        processed_data_dir: str = "processed_data",
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.market_data_path = Path(market_data_path)
        self.regime_data_path = Path(regime_data_path)
        self.processed_data_dir = Path(processed_data_dir)

        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Load model and config
        self.model, self.cfg = self.load_model()

        # Load metadata
        with open(self.processed_data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.seq_length = self.metadata["sequence_length"]

    def load_model(self) -> Tuple[DeepStateSpaceModel, Dict]:
        """Load trained model from checkpoint"""
        print(f"Loading model from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Extract config
        cfg = checkpoint["cfg"]

        # Initialize model
        model = DeepStateSpaceModel(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print("Model loaded successfully")
        print(f"  - Number of states: {cfg.model.num_states}")
        print(f"  - Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(
            f"  - Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}",
        )

        return model, cfg

    def load_market_data(self) -> pd.DataFrame:
        """Load raw market data for plotting"""
        print("Loading market data...")

        # Load market data
        market_df = pd.read_csv(self.market_data_path)

        # Parse timestamp
        if "timestamp_utc" in market_df.columns:
            # Check if it's Unix timestamp or string
            if market_df["timestamp_utc"].dtype == "int64":
                market_df["timestamp"] = pd.to_datetime(
                    market_df["timestamp_utc"],
                    unit="s",
                )
            else:
                market_df["timestamp"] = pd.to_datetime(
                    market_df["timestamp_utc"],
                )

        # Ensure we have required columns
        required_cols = [
            "timestamp",
            "btc_open",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_volume",
        ]
        missing_cols = [
            col for col in required_cols if col not in market_df.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by timestamp
        market_df = market_df.sort_values("timestamp")

        print(f"Loaded {len(market_df)} market data points")
        print(
            f"Date range: {market_df['timestamp'].min()} to {market_df['timestamp'].max()}",
        )

        return market_df

    def load_processed_sequences(self) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """Load all processed sequences for inference"""
        print("Loading processed sequences...")

        all_sequences = []
        all_timestamps = []

        # Load train, val, and test data
        for split in ["train", "val", "test"]:
            data_path = self.processed_data_dir / f"{split}_data.npz"
            if not data_path.exists():
                print(f"Warning: {data_path} not found, skipping...")
                continue

            data = np.load(data_path, allow_pickle=True)
            sequences = data["sequences"]
            timestamps = data["timestamps"]

            all_sequences.append(sequences)
            all_timestamps.extend(timestamps)

            print(f"  - {split}: {len(sequences)} sequences")

        # Concatenate all sequences
        all_sequences = np.concatenate(all_sequences, axis=0)

        # Convert timestamps to datetime
        all_timestamps = pd.to_datetime(all_timestamps)

        print(f"Total sequences loaded: {len(all_sequences)}")

        return all_sequences, all_timestamps

    @torch.no_grad()
    def infer_states(
        self,
        sequences: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Infer states for all sequences"""
        print("Inferring states...")

        self.model.eval()
        all_states = []

        # Process in batches
        n_batches = (len(sequences) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sequences))

            # Get batch
            batch = sequences[start_idx:end_idx]
            batch_tensor = torch.FloatTensor(batch).to(self.device)

            # Infer states
            _, _, pred_states = self.model.compute_elbo(
                batch_tensor,
                return_components=True,
                return_states=True,
            )

            all_states.append(pred_states.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_batches} batches")

        # Concatenate all states
        all_states = np.concatenate(all_states, axis=0)

        print(f"Inferred states for {len(all_states)} sequences")

        # Get state distribution
        unique_states, counts = np.unique(
            all_states.flatten(),
            return_counts=True,
        )
        print("\nState distribution:")
        for state, count in zip(unique_states, counts):
            pct = count / all_states.size * 100
            print(f"  State {state}: {count:,} ({pct:.1f}%)")

        return all_states

    def create_full_timeline(
        self,
        sequences: np.ndarray,
        states: np.ndarray,
        timestamps: pd.DatetimeIndex,
        market_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create full timeline with states aligned to market data"""
        print("Creating full timeline...")

        # Ensure states are on CPU and numpy array
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        # For each sequence, we have states for seq_length hours
        # The timestamp represents the last hour of the sequence
        # We need to expand this to get state for each hour

        hour_states = {}

        for i, (seq_states, end_time) in enumerate(zip(states, timestamps)):
            # Generate timestamps for all hours in this sequence
            seq_timestamps = pd.date_range(
                end=end_time,
                periods=self.seq_length,
                freq="H",
            )

            # Store state for each hour
            for t, state in zip(seq_timestamps, seq_states):
                if t not in hour_states:
                    hour_states[t] = []
                hour_states[t].append(state)

        # Average states for overlapping hours (voting)
        final_states = {}
        for t, state_list in hour_states.items():
            # Use mode (most common state) for each hour
            unique, counts = np.unique(state_list, return_counts=True)
            final_states[t] = unique[np.argmax(counts)]

        # Create dataframe
        state_df = pd.DataFrame(
            list(final_states.items()),
            columns=["timestamp", "state"],
        )
        state_df = state_df.sort_values("timestamp")

        # Merge with market data
        merged_df = pd.merge(market_df, state_df, on="timestamp", how="inner")

        print(f"Created timeline with {len(merged_df)} data points")
        print(
            f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}",
        )

        return merged_df

    def process_month(
        self,
        args: Tuple[pd.DataFrame, int, int, str, Path],
    ) -> Path:
        """Process a single month (for parallel processing)"""
        df_month, year, month, output_prefix, output_dir = args

        if len(df_month) < 2:
            return None

        # Create plotter
        plotter = CryptoStatePlotter(output_dir=output_dir)

        # Create save path
        save_path = output_dir / f"{output_prefix}_{year:04d}_{month:02d}.png"

        # Create and save plot
        plotter.plot_month(df_month, year, month, save_path)

        return save_path

    def plot_all_months_parallel(
        self,
        df: pd.DataFrame,
        output_dir: str = "plots",
        output_prefix: str = "btc_states",
        n_workers: Optional[int] = None,
    ) -> List[Path]:
        """Plot all months in parallel"""
        print("Plotting all months in parallel...")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Extract year and month
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # Prepare arguments for each month
        args_list = []
        for (year, month), df_month in df.groupby(["year", "month"]):
            # Ensure no tensor data in dataframe
            df_month_copy = df_month.copy()
            # Convert any tensor columns to numpy
            for col in df_month_copy.columns:
                if isinstance(df_month_copy[col].iloc[0], torch.Tensor):
                    df_month_copy[col] = df_month_copy[col].apply(
                        lambda x: x.cpu().numpy()
                        if isinstance(x, torch.Tensor)
                        else x,
                    )
            args_list.append(
                (df_month_copy, year, month, output_prefix, output_dir),
            )

        print(
            f"Processing {len(args_list)} months with {n_workers or cpu_count()} workers...",
        )

        # Set multiprocessing start method to spawn for CUDA compatibility
        import multiprocessing as mp

        ctx = mp.get_context("spawn")

        # Process in parallel using spawn context
        with ctx.Pool(n_workers or cpu_count()) as pool:
            saved_paths = pool.map(self.process_month, args_list)

        # Filter out None values
        saved_paths = [p for p in saved_paths if p is not None]

        print(f"Successfully created {len(saved_paths)} plots")

        return saved_paths

    def run_full_pipeline(
        self,
        output_dir: str = "plots",
        batch_size: int = 32,
        n_workers: Optional[int] = None,
    ):
        """Run the full inference and plotting pipeline"""
        print("=" * 60)
        print("Starting Crypto State Inference and Plotting Pipeline")
        print("=" * 60)

        # Load market data
        market_df = self.load_market_data()

        # Load processed sequences
        sequences, timestamps = self.load_processed_sequences()

        # Infer states
        states = self.infer_states(sequences, batch_size)

        # Create full timeline
        full_df = self.create_full_timeline(
            sequences,
            states,
            timestamps,
            market_df,
        )

        # Save the full dataframe for later analysis
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        csv_path = output_path / "inferred_states_full.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"\nSaved full dataset with states to: {csv_path}")

        # Create summary plot
        print("\nCreating summary plot...")
        plotter = CryptoStatePlotter(output_dir=output_dir)
        summary_path = output_path / "btc_states_summary.png"
        plotter.create_summary_plot(full_df, summary_path)

        # Plot all months in parallel
        saved_paths = self.plot_all_months_parallel(
            full_df,
            output_dir=output_dir,
            output_prefix="btc_states",
            n_workers=n_workers,
        )

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Created {len(saved_paths)} monthly plots")
        print(f"Output directory: {output_path}")
        print(f"Summary plot: {summary_path}")
        print(f"Full data CSV: {csv_path}")

        # Print state statistics
        print("\nFinal State Distribution:")
        state_counts = full_df["state"].value_counts().sort_index()
        for state, count in state_counts.items():
            pct = count / len(full_df) * 100
            print(f"  State {state}: {count:,} hours ({pct:.1f}%)")

        return full_df, saved_paths


def main():
    """Main entry point"""
    import argparse
    import multiprocessing as mp

    # Set spawn method for multiprocessing to avoid CUDA fork issues
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description="Infer crypto states and create plots",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_epoch_50.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--market-data",
        type=str,
        default="market_data/merged_market_data_20250216.csv",
        help="Path to market data CSV",
    )
    parser.add_argument(
        "--regime-data",
        type=str,
        default="market_data/market_regime_features_20250216.csv",
        help="Path to regime features CSV",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="processed_data",
        help="Directory with processed sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for plotting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for inference",
    )

    args = parser.parse_args()

    # Create inference object
    inferencer = CryptoStateInference(
        checkpoint_path=args.checkpoint,
        market_data_path=args.market_data,
        regime_data_path=args.regime_data,
        processed_data_dir=args.processed_dir,
        device=args.device,
    )

    # Run pipeline
    inferencer.run_full_pipeline(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
