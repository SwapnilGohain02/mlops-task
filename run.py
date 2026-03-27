"""
MLOps Batch Pipeline - run.py
Trading Signal Generator: Rolling Mean Crossover
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler — full detail
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    required_fields = {"seed": int, "window": int, "version": str}
    for field, expected_type in required_fields.items():
        if field not in config:
            raise KeyError(f"Missing required config field: '{field}'")
        if not isinstance(config[field], expected_type):
            raise TypeError(
                f"Config field '{field}' must be {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )

    if config["window"] < 1:
        raise ValueError(f"Config 'window' must be >= 1, got {config['window']}")

    logger.info(
        f"Config loaded and validated — "
        f"seed={config['seed']}, window={config['window']}, version={config['version']}"
    )
    return config


# ---------------------------------------------------------------------------
# Dataset loading + validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("CSV file contains no data rows.")

    if "close" not in df.columns:
        raise KeyError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if df["close"].isnull().all():
        raise ValueError("Column 'close' contains only null values.")

    logger.info(f"Dataset loaded: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Rolling mean on 'close' with given window.

    NaN handling strategy:
        The first (window - 1) rows will have NaN rolling_mean.
        These rows are EXCLUDED from signal computation and metrics.
        Only rows where rolling_mean is defined are returned.
    """
    logger.debug(f"Computing rolling mean (window={window}) on 'close'...")
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    before = len(df)
    df_valid = df.dropna(subset=["rolling_mean"]).copy()
    dropped = before - len(df_valid)
    logger.debug(
        f"Rolling mean computed. Dropped first {dropped} rows (NaN warm-up). "
        f"Valid rows for signal: {len(df_valid)}"
    )

    logger.debug("Generating binary signal: 1 if close > rolling_mean, else 0 ...")
    df_valid["signal"] = (df_valid["close"] > df_valid["rolling_mean"]).astype(int)

    logger.info(
        f"Signal generation complete — "
        f"rows_processed={len(df_valid)}, "
        f"signal_rate={df_valid['signal'].mean():.6f}"
    )
    return df_valid


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Metrics written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set up logging before anything else
    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps Pipeline — Job START")
    logger.info(f"  input   : {args.input}")
    logger.info(f"  config  : {args.config}")
    logger.info(f"  output  : {args.output}")
    logger.info(f"  log-file: {args.log_file}")
    logger.info("=" * 60)

    job_start = time.time()
    version = "unknown"

    try:
        # --- 1. Load + validate config ---
        config = load_config(args.config, logger)
        version = config["version"]
        seed    = config["seed"]
        window  = config["window"]

        # --- 2. Set seed for reproducibility ---
        np.random.seed(seed)
        logger.debug(f"NumPy random seed set to {seed}")

        # --- 3. Load + validate dataset ---
        df = load_dataset(args.input, logger)

        # --- 4. Compute rolling mean + signal ---
        logger.info("Starting signal computation ...")
        df_result = compute_signals(df, window, logger)

        # --- 5. Compute metrics ---
        rows_processed = len(df_result)
        signal_rate    = float(df_result["signal"].mean())
        latency_ms     = int((time.time() - job_start) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        # --- 6. Write metrics ---
        write_metrics(args.output, metrics, logger)

        # --- 7. Print to stdout (Docker requirement) ---
        print(json.dumps(metrics, indent=2))

        logger.info("=" * 60)
        logger.info(f"Job END — status=success | latency_ms={latency_ms}")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - job_start) * 1000)
        logger.error(f"Pipeline FAILED: {exc}", exc_info=True)

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    latency_ms,
        }

        # Always write metrics — even on failure
        try:
            write_metrics(args.output, error_metrics, logger)
            print(json.dumps(error_metrics, indent=2))
        except Exception as write_exc:
            logger.critical(f"Could not write error metrics: {write_exc}")

        logger.info("=" * 60)
        logger.info(f"Job END — status=error | latency_ms={latency_ms}")
        logger.info("=" * 60)

        sys.exit(1)


if __name__ == "__main__":
    main()