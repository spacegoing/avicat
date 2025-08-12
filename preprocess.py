import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class CryptoDataPreprocessor:
    """Preprocessor for crypto market data to prepare features for DSSM training"""

    def __init__(
        self,
        data_dir: str = "market_data",
        output_dir: str = "processed_data",
        sequence_length: int = 168,  # 7 days of hourly data
        stride: int = 24,  # Step by 1 day
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Feature groups for organized selection
        self.price_features = [
            "btc_open",
            "btc_high",
            "btc_low",
            "btc_close",
            "btc_volume",
        ]

        self.return_features = [
            "btc_return_4h",
            "btc_return_8h",
            "btc_return_24h",
            "btc_return_72h",
            "btc_return_168h",
        ]

        self.technical_features = [
            "rsi_6h",
            "rsi_12h",
            "rsi_24h",
            "macd",
            "macd_signal",
            "macd_hist",
            "volatility_24h",
            "volatility_72h",
            "volatility_168h",
            "atr_24h",
            "atr_72h",
            "atr_168h",
        ]

        self.microstructure_features = [
            "btc_up_ratio_4h",
            "btc_up_ratio_8h",
            "btc_up_ratio_24h",
            "btc_up_ratio_72h",
            "btc_up_ratio_168h",
            "max_up_streak_24h",
            "max_down_streak_24h",
            "up_down_ratio_24h",
            "direction_change_freq_24h",
        ]

        self.sentiment_features = [
            "fear_greed_value",
            "btc_dominance_close",
            "fear_greed_ma_24h",
            "fear_greed_ma_72h",
            "btc_dominance_ma_24h",
            "btc_dominance_ma_72h",
        ]

        self.temporal_features = [
            "hour",
            "day_of_week",
            "asia_session",
            "europe_session",
            "us_session",
        ]

    def load_data(self) -> pd.DataFrame:
        """Load and merge crypto market data"""
        print("Loading market data...")

        # Load main market data
        market_data = pd.read_csv(
            self.data_dir / "merged_market_data_20250216.csv",
        )

        # Load regime features
        regime_features = pd.read_csv(
            self.data_dir / "market_regime_features_20250216.csv",
        )

        # Convert timestamps to datetime
        # Market data has epoch timestamp (integer)
        market_data["timestamp_utc"] = pd.to_datetime(
            market_data["timestamp_utc"],
            unit="s",
        )

        # Regime features has string datetime
        regime_features["timestamp_utc"] = pd.to_datetime(
            regime_features["timestamp_utc"],
        )

        print(f"Market data shape: {market_data.shape}")
        print(f"Regime features shape: {regime_features.shape}")
        print(
            f"Market data timestamp range: {market_data['timestamp_utc'].min()} to {market_data['timestamp_utc'].max()}",
        )
        print(
            f"Regime timestamp range: {regime_features['timestamp_utc'].min()} to {regime_features['timestamp_utc'].max()}",
        )

        # Merge on timestamp
        df = pd.merge(
            market_data,
            regime_features,
            on="timestamp_utc",
            how="inner",
            suffixes=("", "_regime"),
        )

        # Sort by timestamp
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        print(f"Merged data: {len(df)} samples")
        if len(df) > 0:
            print(
                f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}",
            )
        else:
            print(
                "Warning: No matching timestamps found between the two datasets!",
            )
            print("Checking for alignment issues...")
            # Debug: show first few timestamps from each
            print("\nFirst 3 timestamps from market_data:")
            print(market_data["timestamp_utc"].head(3))
            print("\nFirst 3 timestamps from regime_features:")
            print(regime_features["timestamp_utc"].head(3))

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better state representation"""
        print("Engineering features...")

        # Price returns (already in data, but add more granular ones)
        df["btc_return_1h"] = df["btc_close"].pct_change(1)
        df["btc_return_2h"] = df["btc_close"].pct_change(2)

        # Log returns for better statistical properties
        df["btc_log_return_1h"] = np.log(
            df["btc_close"] / df["btc_close"].shift(1),
        )
        df["btc_log_return_4h"] = np.log(
            df["btc_close"] / df["btc_close"].shift(4),
        )

        # Volume features
        df["volume_ma_24h"] = df["btc_volume"].rolling(24).mean()
        df["volume_ratio"] = df["btc_volume"] / df["volume_ma_24h"]

        # Price range
        df["price_range"] = (df["btc_high"] - df["btc_low"]) / df["btc_close"]

        # Momentum indicators
        df["momentum_4h"] = df["btc_close"] - df["btc_close"].shift(4)
        df["momentum_24h"] = df["btc_close"] - df["btc_close"].shift(24)

        # RSI divergence
        df["rsi_divergence"] = df["rsi_24h"] - df["rsi_6h"]

        # MACD momentum
        df["macd_momentum"] = df["macd"] - df["macd_signal"]

        # Volatility ratios
        df["volatility_ratio_24h_72h"] = df["volatility_24h"] / (
            df["volatility_72h"] + 1e-8
        )
        df["volatility_ratio_72h_168h"] = df["volatility_72h"] / (
            df["volatility_168h"] + 1e-8
        )

        # Fear & Greed momentum
        df["fear_greed_momentum"] = (
            df["fear_greed_value"] - df["fear_greed_ma_24h"]
        )

        # Dominance trend
        df["dominance_trend"] = (
            df["btc_dominance_close"] - df["btc_dominance_ma_24h"]
        )

        return df

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Select relevant features for model training"""
        print("Selecting features...")

        # Core features for state-space modeling
        selected_features = []

        # Price action features (normalized returns are key)
        selected_features.extend(
            [
                "btc_log_return_1h",
                "btc_log_return_4h",
                "btc_return_8h",
                "btc_return_24h",
                "btc_return_72h",
            ],
        )

        # Technical indicators
        selected_features.extend(
            [
                "rsi_6h",
                "rsi_12h",
                "rsi_24h",
                "rsi_divergence",
                "macd",
                "macd_signal",
                "macd_momentum",
                "volatility_24h",
                "volatility_72h",
                "volatility_ratio_24h_72h",
            ],
        )

        # Market microstructure
        selected_features.extend(
            [
                "volume_ratio",
                "price_range",
                "btc_up_ratio_24h",
                "up_down_ratio_24h",
                "direction_change_freq_24h",
            ],
        )

        # Sentiment
        selected_features.extend(
            [
                "fear_greed_value",
                "fear_greed_momentum",
                "btc_dominance_close",
                "dominance_trend",
            ],
        )

        # Momentum
        selected_features.extend(
            ["momentum_4h", "momentum_24h", "atr_24h", "atr_ratio_24h"],
        )

        # Temporal (for capturing time-based patterns)
        selected_features.extend(["hour", "day_of_week", "us_session"])

        # Filter to available features
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [
            f for f in selected_features if f not in df.columns
        ]

        if missing_features:
            print(f"Warning: Missing features: {missing_features}")

        print(f"Selected {len(available_features)} features")

        return df[["timestamp_utc"] + available_features], available_features

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> pd.DataFrame:
        """Normalize features using robust scaling"""
        print("Normalizing features...")

        df_normalized = df.copy()

        for col in feature_cols:
            if col in [
                "hour",
                "day_of_week",
                "us_session",
                "asia_session",
                "europe_session",
            ]:
                # Don't normalize categorical/binary features
                continue

            # Use robust scaling (median and IQR) to handle outliers
            median = df[col].median()
            q75 = df[col].quantile(0.75)
            q25 = df[col].quantile(0.25)
            iqr = q75 - q25

            if iqr > 0:
                df_normalized[col] = (df[col] - median) / (iqr + 1e-8)
                # Clip extreme values
                df_normalized[col] = df_normalized[col].clip(-5, 5)
            else:
                df_normalized[col] = 0

        # Fill NaN values
        df_normalized[feature_cols] = df_normalized[feature_cols].fillna(0)

        return df_normalized

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling"""
        print(
            f"Creating sequences (length={self.sequence_length}, stride={self.stride})...",
        )

        sequences = []
        timestamps = []

        for i in range(0, len(df) - self.sequence_length + 1, self.stride):
            seq = df[feature_cols].iloc[i : i + self.sequence_length].values

            # Check for NaN values
            if not np.isnan(seq).any():
                sequences.append(seq)
                timestamps.append(
                    df["timestamp_utc"].iloc[i + self.sequence_length - 1],
                )

        sequences = np.array(sequences, dtype=np.float32)
        timestamps = np.array(timestamps)

        print(
            f"Created {len(sequences)} sequences of shape {sequences[0].shape}",
        )

        return sequences, timestamps

    def split_data(
        self,
        sequences: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Split data into train/val/test sets (time-aware split)"""
        print("Splitting data...")

        n_samples = len(sequences)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Time-based split (no shuffling for time series)
        train_sequences = sequences[:train_end]
        train_timestamps = timestamps[:train_end]

        val_sequences = sequences[train_end:val_end]
        val_timestamps = timestamps[train_end:val_end]

        test_sequences = sequences[val_end:]
        test_timestamps = timestamps[val_end:]

        print(f"Train: {len(train_sequences)} sequences")
        print(f"Val: {len(val_sequences)} sequences")
        print(f"Test: {len(test_sequences)} sequences")

        return (
            train_sequences,
            train_timestamps,
            val_sequences,
            val_timestamps,
            test_sequences,
            test_timestamps,
        )

    def save_processed_data(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        feature_names: list,
        stats: dict,
    ):
        """Save processed data to disk"""
        print("Saving processed data...")

        # Convert timestamps to strings to avoid pickle issues
        train_timestamps = [str(ts) for ts in train_data[1]]
        val_timestamps = [str(ts) for ts in val_data[1]]
        test_timestamps = [str(ts) for ts in test_data[1]]

        np.savez_compressed(
            self.output_dir / "train_data.npz",
            sequences=train_data[0],
            timestamps=np.array(train_timestamps, dtype=str),
        )

        np.savez_compressed(
            self.output_dir / "val_data.npz",
            sequences=val_data[0],
            timestamps=np.array(val_timestamps, dtype=str),
        )

        np.savez_compressed(
            self.output_dir / "test_data.npz",
            sequences=test_data[0],
            timestamps=np.array(test_timestamps, dtype=str),
        )

        # Save metadata
        metadata = {
            "feature_names": feature_names,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "num_features": len(feature_names),
            "stats": stats,
        }

        import json

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Data saved to {self.output_dir}")

    def process(self):
        """Main processing pipeline"""
        print("=" * 60)
        print("Starting Crypto Data Preprocessing")
        print("=" * 60)

        # Load data
        df = self.load_data()

        # Engineer features
        df = self.engineer_features(df)

        # Select features
        df_features, feature_names = self.select_features(df)

        # Calculate statistics before normalization
        stats = {
            "mean": df_features[feature_names].mean().to_dict(),
            "std": df_features[feature_names].std().to_dict(),
            "median": df_features[feature_names].median().to_dict(),
            "q25": df_features[feature_names].quantile(0.25).to_dict(),
            "q75": df_features[feature_names].quantile(0.75).to_dict(),
        }

        # Normalize
        df_normalized = self.normalize_features(df_features, feature_names)

        # Create sequences
        sequences, timestamps = self.create_sequences(
            df_normalized,
            feature_names,
        )

        # Split data
        train_seq, train_ts, val_seq, val_ts, test_seq, test_ts = (
            self.split_data(sequences, timestamps)
        )

        # Save
        self.save_processed_data(
            (train_seq, train_ts),
            (val_seq, val_ts),
            (test_seq, test_ts),
            feature_names,
            stats,
        )

        print("=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)

        return feature_names


if __name__ == "__main__":
    # Configure preprocessing
    preprocessor = CryptoDataPreprocessor(
        data_dir="market_data",
        output_dir="processed_data",
        sequence_length=168,  # 7 days of hourly data
        stride=24,  # Step by 1 day
        train_ratio=0.7,
        val_ratio=0.15,
    )

    # Run preprocessing
    feature_names = preprocessor.process()

    print("\nFeatures used for modeling:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
