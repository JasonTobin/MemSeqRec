"""
Measure per-batch and per-user inference latency for one or more models.

Usage (from repo root):
    python scripts/time_inference.py \
        configs/deezer/baselines/actr_bpr.json \
        configs/deezer/baselines/memseq.json

What is measured
----------------
* Only the model.predict() wall-clock time is measured — data loading,
  feed-dict construction and cache I/O are excluded.
* The first batch is treated as a warm-up (TF graph compilation) and
  excluded from statistics.
* A single cohort seed (1013) is used for both models so results are
  directly comparable.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

# make sure the repo root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from au2actr.utils.params import process_params, gen_model_spec
from au2actr.data.datasets import dataset_factory
from au2actr.models import ModelFactory, UNTRAINED_MODELS
from au2actr.data.loaders import dataloader_factory
from au2actr.logging import get_logger

SEED = 1013  # single cohort seed for timing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feed_dict_for_batch(batch_data, dataloader, model, model_name):
    """Replicate the feed-dict logic from commands/eval.py."""
    feed_dict = {}
    feed_dict["model_feed"] = model.build_feedict(batch_data, is_training=False)

    if "top" in model_name:
        feed_dict["user_ids"] = batch_data[0]
        feed_dict["item_pops"] = dataloader.get_item_pops()
        feed_dict["n_test"] = dataloader.get_num_test_sessions()
    else:
        feed_dict["item_ids"] = dataloader.item_ids
        feed_dict["user_ids"] = batch_data[-1]

    if model_name == "actr":
        feed_dict["nxt_indices"] = batch_data[1]
        actr_scores = defaultdict(dict)
        for idx in range(len(batch_data[-1])):
            uid = batch_data[-1][idx]
            nxt_idx = batch_data[1][idx]
            actr = batch_data[-2][idx]
            actr_scores[uid][nxt_idx] = actr
        model.actr_scores = actr_scores

    if model_name == "actr_bpr":
        feed_dict["nxt_indices"] = batch_data[1]
        actr_scores = []
        for idx in range(len(batch_data[-1])):
            uid = batch_data[-1][idx]
            nxt_idx = batch_data[1][idx]
            actr = dataloader.data["test_actr_scores"][uid][nxt_idx].toarray()[0]
            actr_scores.append(actr)
        feed_dict["actr_scores"] = np.array(actr_scores)

    return feed_dict


def _time_model(config_path, top_n=10):
    """Load model from config_path and time inference over one cohort."""
    logger = get_logger()
    tf.compat.v1.reset_default_graph()

    with open(config_path) as f:
        params = json.load(f)

    params["command"] = "eval"
    training_params, model_params = process_params(params)
    dataset_params = params["dataset"]
    eval_params = params["eval"]
    min_sessions = dataset_params.get("min_sessions", 250)
    embedding_dim = training_params.get("embedding_dim", 128)
    model_name = training_params["model"]["name"]
    batch_size = eval_params.get("batch_size", 256)
    seqlen = model_params.get("seqlen", 20)
    num_favs = model_params.get("num_favs", 0)

    logger.info(f"Loading dataset for {model_name} ...")
    data = dataset_factory(params=params)
    pretrained_embs = {
        "item_ids": np.array(list(data["svd_embeddings"].keys())),
        "svd_embeddings": np.array(list(data["svd_embeddings"].values())),
        "audio_embeddings": np.array(list(data["audio_embeddings"].values())),
    }

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with tf.compat.v1.Session(config=sess_config) as sess:
        model = ModelFactory.generate_model(
            sess=sess,
            params=training_params,
            n_users=data["n_users"],
            n_items=data["n_items"],
            command="eval",
            pretrained_embs=pretrained_embs,
        )
        if "last" in model_name:
            sess.run(tf.compat.v1.global_variables_initializer())

        if model_name == "actr":
            model.set_user_tracks(data["user_tracks"]["train"])
            model.set_item_ids_map(
                {iid: idx for idx, iid in enumerate(data["track_ids"])}
            )
        if model_name == "actr_bpr":
            model.activate_actr = True

        dataloader = dataloader_factory(
            data=data,
            batch_size=batch_size,
            seqlen=seqlen,
            mode="test",
            num_scored_users=eval_params.get("n_users", -1),
            model_name=model_name,
            embedding_dim=embedding_dim,
            random_seed=SEED,
            num_favs=num_favs,
            command="eval",
            aggregate_type=None,
            activate_actr=True,
        )

        n_batches = dataloader.get_num_batches()
        n_users_total = 0
        batch_times = []

        logger.info(
            f"Timing {model_name} over {n_batches} batches "
            f"(batch 1 = warm-up, excluded from stats) ..."
        )

        for b_idx in range(1, n_batches):
            batch_data = dataloader.next_batch()
            feed_dict = _build_feed_dict_for_batch(
                batch_data, dataloader, model, model_name
            )

            t0 = time.perf_counter()
            batch_reco = model.predict(feed_dict, top_n=top_n)
            t1 = time.perf_counter()

            n_users_in_batch = len(batch_reco)

            if b_idx == 1:
                # warm-up: TF graph compilation happens here; skip
                logger.info(
                    f"  Batch 1 (warm-up): {(t1-t0)*1000:.1f} ms — excluded"
                )
                continue

            batch_times.append(t1 - t0)
            n_users_total += n_users_in_batch

        batch_times = np.array(batch_times)
        total_s = batch_times.sum()
        mean_batch_ms = batch_times.mean() * 1000
        std_batch_ms  = batch_times.std()  * 1000
        p50_ms = np.percentile(batch_times, 50) * 1000
        p95_ms = np.percentile(batch_times, 95) * 1000
        per_user_ms = (total_s / n_users_total) * 1000 if n_users_total else 0

        return {
            "model": model_name,
            "config": config_path,
            "n_batches_timed": len(batch_times),
            "n_users_timed": n_users_total,
            "total_s": total_s,
            "mean_batch_ms": mean_batch_ms,
            "std_batch_ms": std_batch_ms,
            "p50_batch_ms": p50_ms,
            "p95_batch_ms": p95_ms,
            "per_user_ms": per_user_ms,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument(
        "configs",
        nargs="+",
        help="One or more config JSON paths",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N recommendation cut-off (default: 10)",
    )
    args = parser.parse_args()

    tf.compat.v1.disable_eager_execution()

    results = []
    for cfg in args.configs:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg}")
        print(f"{'='*60}")
        r = _time_model(cfg, top_n=args.top_n)
        results.append(r)

    # ── Summary table ──────────────────────────────────────────────
    col = 20
    print(f"\n{'='*60}")
    print("  INFERENCE LATENCY SUMMARY  (warm-up batch excluded)")
    print(f"{'='*60}")
    header = f"{'Model':<{col}} {'Total(s)':>9} {'Mean bat(ms)':>13} "  \
             f"{'Std(ms)':>9} {'P50(ms)':>9} {'P95(ms)':>9} {'Per user(ms)':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<{col}} "
            f"{r['total_s']:>9.3f} "
            f"{r['mean_batch_ms']:>13.2f} "
            f"{r['std_batch_ms']:>9.2f} "
            f"{r['p50_batch_ms']:>9.2f} "
            f"{r['p95_batch_ms']:>9.2f} "
            f"{r['per_user_ms']:>13.4f}"
        )
    print(f"{'='*60}")
    print(f"Cohort seed: {SEED}  |  top_n: {args.top_n}")


if __name__ == "__main__":
    main()
