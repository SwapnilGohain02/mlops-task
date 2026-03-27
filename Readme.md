# MLOps Task 0 — Trading Signal Pipeline

A minimal MLOps-style batch job that loads OHLCV market data, computes a rolling mean on `close` price, generates a binary trading signal, and outputs structured metrics with full logging.

---

## Directory Structure

```
mlops-task/
├── run.py            # Main pipeline script
├── config.yaml       # Configuration (seed, window, version)
├── data.csv          # Input OHLCV dataset (10,000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── README.md         # This file
├── metrics.json      # Sample output from a successful run
└── run.log           # Sample log from a successful run
```

---

## Signal Logic

```
rolling_mean[i] = mean(close[i-window+1 : i+1])
signal[i]       = 1  if close[i] > rolling_mean[i]
                  0  otherwise
```

**NaN handling:** The first `window - 1` rows have no complete window and are excluded from signal computation. `rows_processed` reflects only valid rows.

---

## Local Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four arguments are required — no paths are hardcoded.

---

## Docker Build & Run

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline, prints metrics JSON to stdout, and exits with code `0` on success or `1` on failure.

### Retrieve output files from the container (optional)

```bash
# Run with a mounted output directory
docker run --rm -v $(pwd)/output:/app/output mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   output/metrics.json \
    --log-file output/run.log
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 57,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` is 9996 (not 10000) because the first 4 rows (window - 1 = 4) lack a complete rolling window and are excluded.

### Error output format

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong",
  "latency_ms": 12
}
```

Metrics are written in **both** success and error cases.

---

## Config Reference

| Field     | Type   | Description                          |
|-----------|--------|--------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility|
| `window`  | int    | Rolling mean window size (rows)      |
| `version` | string | Pipeline version tag in output JSON  |

---

## Reproducibility

The pipeline is fully deterministic:
- `numpy.random.seed(seed)` is called at startup
- All computation is pure pandas rolling operations
- Same config + same data → identical `metrics.json` every run

---

## Error Handling

The pipeline handles and logs the following failure modes:

| Condition                        | Behaviour                              |
|----------------------------------|----------------------------------------|
| Missing input file               | Error metrics written, exit code 1     |
| Empty or unparseable CSV         | Error metrics written, exit code 1     |
| Missing `close` column           | Error metrics written, exit code 1     |
| Missing config file              | Error metrics written, exit code 1     |
| Missing required config field    | Error metrics written, exit code 1     |
| Wrong config field type          | Error metrics written, exit code 1     |