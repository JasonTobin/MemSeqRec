"""
Preprocess the raw Deezer-RecSys25 tarball into the format expected by the
au2actr codebase.

What this script does
─────────────────────
Pass 1  — Sequential scan of the tarball (one read, O(n) decompression):
  • track_embeddings shards (only 2) → concatenated and written directly.
  • Session shards: read user_id + session_id (2 cols), dedup within shard,
    write as a small temp Parquet file (~10–30 MB each) AND accumulate
    per-shard unique-session COUNTS per user (just ints, ~few hundred MB RAM).

Qualify — Two-step, memory-efficient filter:
  Step A  Upper-bound filter: users whose sum of per-shard counts < min_sessions
          cannot possibly qualify. Discard them. (O(n_users) RAM, negligible.)
  Step B  Exact dedup: re-read temp files but only for the small "candidate"
          set that survived step A. Use numpy arrays for compact storage.
          Memory ≈ n_candidates × avg_sessions_per_candidate × 8 bytes.

Pass 2  — Sequential scan of the tarball a second time:
  • Session shards filtered to qualifying users, written to one Parquet file.

WHY sequential iteration, not getmembers() + extractfile()?
  tarfile r:gz wraps gzip which is NOT randomly seekable. getmembers() must
  decompress the whole archive once. Any subsequent extractfile() call on a
  stored TarInfo then has to re-decompress from the start to reach that
  member's offset — O(n²) total decompression work on a 9.8 GB file. The
  `for member in tar:` iterator reads each member exactly once: O(n).

Expected output layout (matches all configs out-of-the-box):
  <data-dir>/deezer/min<N>sess/sessions          ← filtered sessions Parquet
  <data-dir>/deezer/min<N>sess/track_embeddings  ← full track embeddings Parquet

Column names in the tarball match exactly what the code expects:
  sessions:          user_id, track_id, session_id, ts
  track_embeddings:  track_id, art_id, svd, audio

Usage (run from the repo root):
  python scripts/prepare_data.py
  python scripts/prepare_data.py --tar deezer-recsys25.tar.gz \\
      --data-dir exp/data --min-sessions 300
"""

import argparse
import io
import os
import shutil
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess deezer-recsys25.tar.gz for au2actr"
    )
    parser.add_argument(
        "--tar",
        default="deezer-recsys25.tar.gz",
        help="Path to the raw Deezer tarball (default: deezer-recsys25.tar.gz)",
    )
    parser.add_argument(
        "--data-dir",
        default="exp/data",
        help="Root data directory expected by configs (default: exp/data)",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=300,
        help="Minimum number of distinct sessions a user must have (default: 300)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1
# ──────────────────────────────────────────────────────────────────────────────

def pass1_sequential(tar_path, temp_dir, track_embs_out):
    """
    One sequential pass through the tarball.

    Returns
    -------
    user_approx_counts : dict[int, int]
        Upper bound on each user's distinct session count (sum of per-shard
        nunique values; overcounts when the same session spans shards, but
        never undercounts).
    n_session_shards : int
    """
    track_chunks = []
    shard_idx = 0
    user_approx_counts = defaultdict(int)

    os.makedirs(temp_dir, exist_ok=True)

    print(f"Pass 1 — sequential scan of {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read()

            # ── track embeddings ──────────────────────────────────────────────
            if "/track_embeddings/" in member.name and not os.path.exists(track_embs_out):
                print(f"  [tracks ] {member.name}")
                track_chunks.append(pd.read_parquet(io.BytesIO(raw)))

            # ── session shards ────────────────────────────────────────────────
            elif "/user_sessions/" in member.name:
                if shard_idx % 100 == 0:
                    print(f"  [sessions] shard {shard_idx:>4d} — {member.name}")
                df = pd.read_parquet(io.BytesIO(raw), columns=["user_id", "session_id"])
                df.drop_duplicates(inplace=True)
                # Accumulate upper-bound counts (fast, negligible memory)
                per_shard = df.groupby("user_id", sort=False)["session_id"].nunique()
                for uid, cnt in per_shard.items():
                    user_approx_counts[uid] += cnt
                # Save deduped pairs for the exact-count step
                df.to_parquet(
                    os.path.join(temp_dir, f"uis_{shard_idx:06d}.parquet"),
                    index=False,
                )
                shard_idx += 1

    # Write track embeddings collected above
    if track_chunks and not os.path.exists(track_embs_out):
        track_df = pd.concat(track_chunks, ignore_index=True)
        track_df.drop_duplicates(subset=["track_id"], inplace=True)
        print(f"  Total unique tracks: {len(track_df):,}")
        track_df.to_parquet(track_embs_out, index=False)
        print(f"  Track embeddings → {track_embs_out}")

    print(f"  Pass 1 done — {shard_idx} session shards, {len(user_approx_counts):,} users.")
    return dict(user_approx_counts), shard_idx


# ──────────────────────────────────────────────────────────────────────────────
# Qualify
# ──────────────────────────────────────────────────────────────────────────────

def compute_qualifying_users(temp_dir, user_approx_counts, min_sessions):
    """
    Two-step qualification to keep RAM usage manageable even when users are
    spread across hundreds of shards.

    Step A — upper-bound pre-filter
    --------------------------------
    Users whose sum-of-per-shard nunique < min_sessions cannot qualify.
    This leaves a small "candidate" set whose exact count we need to verify.

    Step B — exact dedup for candidates only
    ----------------------------------------
    Re-read the temp files (small: deduped within each shard), but filter
    immediately to candidates. Accumulate session IDs as numpy int64 arrays
    (8 bytes/entry, much cheaper than Python sets or dicts).
    """
    # Step A ─────────────────────────────────────────────────────────────────
    candidates = frozenset(
        uid for uid, cnt in user_approx_counts.items() if cnt >= min_sessions
    )
    n_total = len(user_approx_counts)
    print(
        f"Qualify step A — upper-bound filter: "
        f"{len(candidates):,} candidates / {n_total:,} users "
        f"(approx. sum >= {min_sessions})"
    )

    # Step B ─────────────────────────────────────────────────────────────────
    # Compact numpy-backed storage: user_id → sorted int64 array of session IDs
    import glob
    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))

    candidate_sids: dict[int, np.ndarray] = {}  # user_id → sorted unique sids

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


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2
# ──────────────────────────────────────────────────────────────────────────────

def pass2_filter_sessions(tar_path, qualifying_users, sessions_out):
    """Second sequential pass: filter sessions to qualifying users."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Track embeddings (resume path — only needed if sessions were already done)
# ──────────────────────────────────────────────────────────────────────────────

def collect_track_embeddings(tar_path, track_embs_out):
    """Collect track-embedding shards from the tarball and write one Parquet."""
    print(f"Collecting track embeddings from {tar_path} ...")
    chunks = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isdir() or "/track_embeddings/" not in member.name:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            print(f"  {member.name}")
            chunks.append(pd.read_parquet(io.BytesIO(f.read())))
    track_df = pd.concat(chunks, ignore_index=True)
    track_df.drop_duplicates(subset=["track_id"], inplace=True)
    track_df.to_parquet(track_embs_out, index=False)
    print(f"  Track embeddings → {track_embs_out}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir = os.path.join(args.data_dir, "deezer", f"min{args.min_sessions}sess")
    sessions_out = os.path.join(out_dir, "sessions")
    track_embs_out = os.path.join(out_dir, "track_embeddings")
    temp_dir = os.path.join(out_dir, "_tmp_uid_sid")

    os.makedirs(out_dir, exist_ok=True)

    sessions_done = os.path.exists(sessions_out)
    track_done = os.path.exists(track_embs_out)

    if sessions_done and track_done:
        print("Both output files already exist — nothing to do.")
        print(f"  {sessions_out}")
        print(f"  {track_embs_out}")
        return

    if not sessions_done:
        # ── Pass 1: write temp files + approx counts + track embeddings ───────
        user_approx_counts, _ = pass1_sequential(args.tar, temp_dir, track_embs_out)

        # ── Qualify users ─────────────────────────────────────────────────────
        qualifying_users = compute_qualifying_users(
            temp_dir, user_approx_counts, args.min_sessions
        )
        del user_approx_counts

        # Temp files no longer needed
        shutil.rmtree(temp_dir, ignore_errors=True)

        # ── Pass 2: filter and write sessions ─────────────────────────────────
        pass2_filter_sessions(args.tar, qualifying_users, sessions_out)
    else:
        print(f"Sessions already written — skipping: {sessions_out}")

    # Resume path: sessions done but track embeddings missing
    if not track_done:
        collect_track_embeddings(args.tar, track_embs_out)

    print("\nPreprocessing complete.")
    print(f"  Sessions:          {sessions_out}")
    print(f"  Track embeddings:  {track_embs_out}")
    print(
        f"\nVerify configs have:\n"
        f'  "dataset": {{\n'
        f'    "path": "{args.data_dir}",\n'
        f'    "name": "deezer",\n'
        f'    "min_sessions": {args.min_sessions},\n'
        f'    ...\n'
        f"  }}"
    )


if __name__ == "__main__":
    main()
