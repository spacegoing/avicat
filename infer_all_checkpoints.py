import argparse
import json
import multiprocessing as mp
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from plot_crypto_states import CryptoStatePlotter

from model import DeepStateSpaceModel


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

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model, self.cfg = self.load_model()
        with open(self.processed_data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]
        self.seq_length = self.metadata["sequence_length"]

    def load_model(self) -> Tuple[DeepStateSpaceModel, Dict]:
        print(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        cfg = checkpoint["cfg"]
        model = DeepStateSpaceModel(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        print("Model loaded successfully")
        return model, cfg

    def load_market_data(self) -> pd.DataFrame:
        print("Loading market data...")
        market_df = pd.read_csv(self.market_data_path)
        if "timestamp_utc" in market_df.columns:
            if market_df["timestamp_utc"].dtype == "int64":
                market_df["timestamp"] = pd.to_datetime(
                    market_df["timestamp_utc"],
                    unit="s",
                )
            else:
                market_df["timestamp"] = pd.to_datetime(
                    market_df["timestamp_utc"],
                )
        required_cols = [
            "timestamp",
            "btc_open",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_volume",
        ]
        if any(col not in market_df.columns for col in required_cols):
            raise ValueError("Missing required columns")
        market_df = market_df.sort_values("timestamp")
        return market_df

    def load_processed_sequences(self) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        print("Loading processed sequences...")
        all_sequences, all_timestamps = [], []
        for split in ["train", "val", "test"]:
            data_path = self.processed_data_dir / f"{split}_data.npz"
            if data_path.exists():
                data = np.load(data_path, allow_pickle=True)
                all_sequences.append(data["sequences"])
                all_timestamps.extend(data["timestamps"])
        return np.concatenate(all_sequences, axis=0), pd.to_datetime(
            all_timestamps,
        )

    @torch.no_grad()
    def infer_states(
        self,
        sequences: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        print("Inferring states...")
        self.model.eval()
        all_states = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        for i in range(n_batches):
            batch = sequences[i * batch_size : (i + 1) * batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            _, _, pred_states = self.model.compute_elbo(
                batch_tensor,
                return_components=True,
                return_states=True,
            )
            all_states.append(pred_states.cpu().numpy())
        return np.concatenate(all_states, axis=0)

    def create_full_timeline(
        self,
        states: np.ndarray,
        timestamps: pd.DatetimeIndex,
        market_df: pd.DataFrame,
    ) -> pd.DataFrame:
        print("Creating full timeline...")
        hour_states = {}
        for seq_states, end_time in zip(states, timestamps):
            seq_timestamps = pd.date_range(
                end=end_time,
                periods=self.seq_length,
                freq="H",
            )
            for t, state in zip(seq_timestamps, seq_states):
                if t not in hour_states:
                    hour_states[t] = []
                hour_states[t].append(state)
        final_states = {
            t: np.bincount(state_list).argmax()
            for t, state_list in hour_states.items()
        }
        state_df = pd.DataFrame(
            list(final_states.items()),
            columns=["timestamp", "state"],
        )
        return pd.merge(
            market_df,
            state_df,
            on="timestamp",
            how="inner",
        ).sort_values("timestamp")

    def run_full_pipeline_for_checkpoint(
        self,
        output_dir: Path,
        batch_size: int,
        n_workers: Optional[int],
    ):
        print("=" * 60)
        print(f"Processing checkpoint: {self.checkpoint_path.name}")
        print("=" * 60)

        market_df = self.load_market_data()
        sequences, timestamps = self.load_processed_sequences()
        states = self.infer_states(sequences, batch_size)
        full_df = self.create_full_timeline(states, timestamps, market_df)

        output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = output_dir / "inferred_states_full.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"\nSaved full dataset with states to: {csv_path}")

        plotter = CryptoStatePlotter(output_dir=str(output_dir))
        summary_path = output_dir / "btc_states_summary.html"
        plotter.create_summary_plot(full_df, summary_path)

        # This part remains the same as it calls the modified plotter
        full_df["year"] = full_df["timestamp"].dt.year
        full_df["month"] = full_df["timestamp"].dt.month
        args_list = [
            (df_month.copy(), year, month, "btc_states", output_dir)
            for (year, month), df_month in full_df.groupby(["year", "month"])
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers or cpu_count()) as pool:
            pool.map(self.process_month_for_plotter, args_list)

        print("\n" + "=" * 60)
        print(f"Pipeline Complete for {self.checkpoint_path.name}!")
        print("=" * 60)

    def process_month_for_plotter(self, args):
        df_month, year, month, output_prefix, output_dir = args
        plotter = CryptoStatePlotter(output_dir=str(output_dir))
        save_path = output_dir / f"{output_prefix}_{year:04d}_{month:02d}.html"
        plotter.plot_month(df_month, year, month, save_path)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Infer crypto states for ALL checkpoints.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory with model checkpoints",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="analysis_results",
        help="Base output directory for all analyses",
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

    ckpt_dir = Path(args.checkpoints_dir)
    checkpoint_files = sorted(list(ckpt_dir.glob("*.pt")))

    if not checkpoint_files:
        print(f"No checkpoint files (.pt) found in '{ckpt_dir}'")
        return

    print(f"Found {len(checkpoint_files)} checkpoints to analyze.")

    for ckpt_file in checkpoint_files:
        output_dir = Path(args.output_base_dir) / ckpt_file.stem
        inferencer = CryptoStateInference(
            checkpoint_path=str(ckpt_file),
            device=args.device,
        )
        inferencer.run_full_pipeline_for_checkpoint(
            output_dir=output_dir,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
        )


if __name__ == "__main__":
    main()
