"""
Resume data preparation after a crash that left Pass 1 temp files intact.

Skips Pass 1 entirely by reconstructing user_approx_counts from the existing
_tmp_uid_sid/ parquet files, then runs Qualify + Pass 2 (one tarball scan).

Usage (run from the repo root):
  python scripts/resume_prepare_data.py
  python scripts/resume_prepare_data.py --tar deezer-recsys25.tar.gz \
      --data-dir exp/data --min-sessions 300
"""

import argparse
import glob
import io
import os
import shutil
import tarfile

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume prepare_data.py after a crash (temp files already exist)"
    )
    parser.add_argument("--tar", default="deezer-recsys25.tar.gz")
    parser.add_argument("--data-dir", default="exp/data")
    parser.add_argument("--min-sessions", type=int, default=300)
    return parser.parse_args()


def rebuild_approx_counts(temp_dir):
    """Reconstruct user_approx_counts from existing _tmp_uid_sid/ parquets."""
    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))
    if not temp_files:
        raise FileNotFoundError(f"No temp parquets found in {temp_dir}")
    print(f"Rebuilding approx counts from {len(temp_files)} existing temp files ...")
    from collections import defaultdict
    user_approx_counts = defaultdict(int)
    for i, fp in enumerate(temp_files):
        if i % 100 == 0:
            print(f"  temp file {i:>4d}/{len(temp_files)} ...")
        df = pd.read_parquet(fp)
        per_shard = df.groupby("user_id", sort=False)["session_id"].nunique()
        for uid, cnt in per_shard.items():
            user_approx_counts[uid] += cnt
    print(f"  Done — {len(user_approx_counts):,} users.")
    return dict(user_approx_counts)


def compute_qualifying_users(temp_dir, user_approx_counts, min_sessions):
    candidates = frozenset(
        uid for uid, cnt in user_approx_counts.items() if cnt >= min_sessions
    )
    n_total = len(user_approx_counts)
    print(
        f"Qualify step A — upper-bound filter: "
        f"{len(candidates):,} candidates / {n_total:,} users "
        f"(approx. sum >= {min_sessions})"
    )

    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))
    candidate_sids: dict = {}

    print(f"Qualify step B — exact dedup across {len(temp_files)} temp files ...")
    for i, fp in enumerate(temp_files):
        if i % 100 == 0:
            print(f"  temp file {i:>4d}/{len(temp_files)} ...")
        df = pd.read_parquet(fp)
        df = df[df["user_id"].isin(candidates)]
        if df.empty:
            continue
        for uid, grp in df.groupby("user_id", sort=False):
            new_sids = np.unique(grp["session_id"].to_numpy(dtype=np.int64))
            if uid in candidate_sids:
                candidate_sids[uid] = np.unique(
                    np.concatenate([candidate_sids[uid], new_sids])
                )
            else:
                candidate_sids[uid] = new_sids

    qualifying = frozenset(
        uid for uid, arr in candidate_sids.items() if len(arr) >= min_sessions
    )
    print(
        f"  Qualifying users (>= {min_sessions} distinct sessions): "
        f"{len(qualifying):,} / {n_total:,}"
    )
    return qualifying


def pass2_filter_sessions(tar_path, qualifying_users, sessions_out):
    print(f"Pass 2 — filtering sessions to {len(qualifying_users):,} users ...")
    chunks = []
    shard_idx = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isdir() or "/user_sessions/" not in member.name:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            if shard_idx % 100 == 0:
                print(f"  shard {shard_idx:>4d} — {member.name}")
            raw = f.read()
            df = pd.read_parquet(io.BytesIO(raw))
            filtered = df[df["user_id"].isin(qualifying_users)]
            if len(filtered) > 0:
                chunks.append(filtered)
            shard_idx += 1

    sessions_df = pd.concat(chunks, ignore_index=True)
    print(f"  Total filtered rows: {len(sessions_df):,}")
    sessions_df.to_parquet(sessions_out, index=False)
    print(f"  Sessions → {sessions_out}")


def main():
    args = parse_args()

    out_dir = os.path.join(args.data_dir, "deezer", f"min{args.min_sessions}sess")
    sessions_out = os.path.join(out_dir, "sessions")
    temp_dir = os.path.join(out_dir, "_tmp_uid_sid")

    if os.path.exists(sessions_out):
        print(f"Sessions already exist at {sessions_out} — nothing to do.")
        return

    # Rebuild approx counts from existing temp files (skips tarball Pass 1)
    user_approx_counts = rebuild_approx_counts(temp_dir)

    qualifying_users = compute_qualifying_users(
        temp_dir, user_approx_counts, args.min_sessions
    )
    del user_approx_counts

    # Temp files no longer needed
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Cleaned up {temp_dir}")

    pass2_filter_sessions(args.tar, qualifying_users, sessions_out)

    print("\nPreprocessing complete.")
    print(f"  Sessions: {sessions_out}")


if __name__ == "__main__":
    main()
