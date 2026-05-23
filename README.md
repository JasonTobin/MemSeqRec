# "MemSeqRec: A Cognitive-Attentive Sequential Recommender
with ANN Re-Ranking for Music Streaming

Video Presentation:
https://www.youtube.com/watch?v=RtBZ0q41WwA

1. [Environment](#environment)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [Running Models](#running-models)
5. [Hyperparameters](#hyperparameters)

---

## Environment

Install the following dependencies:

```
python      3.9.13
tensorflow  2.11.0
tqdm        4.65.0
numpy       1.24.2
scipy       1.10.1
pandas      1.5.3
toolz       0.12.0
```
## Dataset

The anonymized Deezer-RecSys25 dataset is freely available on
[Zenodo](https://zenodo.org/records/17154089).
It contains ~900 million time-stamped listening events from 4 million users (2023),
covering 50,000 anonymized tracks with Audio and SVD multimodal embeddings.
All files are in Parquet format.

Download the tarball (`deezer-recsys25.tar.gz`) and place it in the repo root, or
pass its path explicitly via `--tar`

---

## Data Preparation

Run the preparation script **once** before any training or evaluation.
It makes two sequential passes through the tarball, filters users to those with at
least 300 distinct sessions, and writes the processed files to `exp/data/`.

Be prepared for this to take ~30GB of space

```bash
# Default: reads ./deezer-recsys25.tar.gz, writes to exp/data/, min 300 sessions
python scripts/prepare_data.py

# Explicit paths / threshold
python scripts/prepare_data.py \
    --tar path/to/deezer-recsys25.tar.gz \
    --data-dir exp/data \
    --min-sessions 300
```

**Output layout** (matches all config files out-of-the-box):

```
exp/data/
└── deezer/
    └── min300sess/
        ├── sessions          <- filtered listening sessions (Parquet)
        └── track_embeddings  <- track Audio + SVD embeddings (Parquet)
```

The first time a model is trained or evaluated, derived cache files (BLL weights,
session indexes, spread weights, etc.) are built automatically and stored under
`cache/deezer/min300sess/`. This takes several minutes but only happens once.

---

## Running Models

All models share the same CLI entry point:

```bash
python -m au2actr train --verbose -p <config>   # train a model
python -m au2actr eval  --verbose -p <config>   # evaluate a trained model
```

Configs live in `configs/deezer/` and `configs/deezer/baselines/`.
Trained checkpoints are saved under `exp/model/`.

---

### ACT-R (heuristic baseline — no training required)

ACT-R is a parameter-free cognitive model; only evaluation is needed.

```bash
python -m au2actr eval --verbose -p configs/deezer/baselines/actr.json
```

---

### MemSeqRec (Proposed Model)

```bash
python -m au2actr train --verbose -p configs/deezer/baselines/memseq.json
python -m au2actr eval  --verbose -p configs/deezer/baselines/memseq.json
```

---

### Measuring inference latency

To compare per-user inference time across models:

```bash
python scripts/time_inference.py \
    configs/deezer/baselines/actr_bpr.json \
    configs/deezer/baselines/memseq.json
```

Pass any number of config paths. The script reports mean, P50, and P95 batch
latency and per-user latency for each model (warm-up batch excluded).

---

## Hyperparameters

Hyperparameters for each model are in the corresponding config file under `configs/`.
Shared defaults across all models:

| Parameter                            | Value      |
| ------------------------------------ | ---------- |
| Epochs                               | 100        |
| Optimizer                            | Adam       |
| Batch size                           | 512        |
| Embedding dim                        | 128        |
| ACT-R alpha                          | 0.5        |
| Transformer blocks / heads / seq len | 2 / 2 / 30 |

Hyperparameters tuned via grid search on the validation set:

- Learning rate: {0.0002, 0.0005, 0.00075, 0.001}
- lambda: {0.0, 0.3, 0.5, 0.8, 0.9, 1.0}
- beta and gamma: {0.2, 0.4, 0.6, 0.8, 1.0}
