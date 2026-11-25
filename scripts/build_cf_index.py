"""Offline builder that compresses the 399MB BGG user_ratings.csv into a lightweight
Top-K item-based collaborative filtering index (data/cf_topk.json.gz) for low-memory
cloud deployments (<250MB RAM, <2s startup).
"""

import gzip
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_cf_index(
    bgg_csv: str,
    ratings_csv: str,
    out_gz: str,
    min_user_ratings: int = 20,
    min_game_ratings: int = 50,
    top_k: int = 30,
):
    t0 = time.time()
    bgg = pd.read_csv(bgg_csv, encoding="latin1")
    valid_ids = set(bgg["ID"].dropna().astype(int).tolist())
    print(f"Valid BGG IDs: {len(valid_ids):,}")

    chunks = []
    for chunk in pd.read_csv(
        ratings_csv, chunksize=500_000, usecols=["Username", "BGGId", "Rating"]
    ):
        chunk = chunk.dropna()
        chunk = chunk[(chunk["Rating"] > 0) & (chunk["BGGId"].isin(valid_ids))]
        if len(chunk) > 0:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"Loaded {len(df):,} ratings in {time.time() - t0:.1f}s")

    u_cnt = df["Username"].value_counts()
    g_cnt = df["BGGId"].value_counts()
    df = df[
        df["Username"].isin(u_cnt[u_cnt >= min_user_ratings].index)
        & df["BGGId"].isin(g_cnt[g_cnt >= min_game_ratings].index)
    ]
    n_users = df["Username"].nunique()
    n_games = df["BGGId"].nunique()
    print(f"Filtered to {len(df):,} ratings | users={n_users:,} | games={n_games:,}")

    stats = df.groupby("BGGId")["Rating"].agg(["mean", "count"])
    avg_ratings = {
        int(gid): {
            "avg_rating": round(float(r["mean"]), 2),
            "rating_count": int(r["count"]),
        }
        for gid, r in stats.iterrows()
    }
    popular_games = [
        int(x) for x in stats.sort_values("count", ascending=False).head(100).index
    ]

    unique_users = df["Username"].unique()
    unique_games = df["BGGId"].unique()
    u_map = pd.Series(np.arange(len(unique_users), dtype=np.int32), index=unique_users)
    g_map = pd.Series(np.arange(len(unique_games), dtype=np.int32), index=unique_games)

    u_idx = df["Username"].map(u_map).values
    g_idx = df["BGGId"].map(g_map).values
    vals = df["Rating"].values.astype(np.float32)

    item_user = csr_matrix(
        (vals, (g_idx, u_idx)),
        shape=(len(unique_games), len(unique_users)),
        dtype=np.float32,
    )
    del df, u_idx, g_idx, vals

    neighbors = {}
    n_items = item_user.shape[0]
    chunk_sz = 500
    for start in range(0, n_items, chunk_sz):
        end = min(start + chunk_sz, n_items)
        sim = cosine_similarity(item_user[start:end], item_user).astype(np.float32)
        for local_i in range(end - start):
            global_i = start + local_i
            row = sim[local_i]
            row[global_i] = 0.0
            k = min(top_k, n_items - 1)
            top_idx = np.argpartition(row, -k)[-k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            pairs = [
                [int(unique_games[j]), round(float(row[j]), 4)]
                for j in top_idx
                if row[j] > 0.1
            ]
            if pairs:
                neighbors[int(unique_games[global_i])] = pairs

    os.makedirs(os.path.dirname(out_gz), exist_ok=True)
    payload = {
        "neighbors": neighbors,
        "avg_ratings": avg_ratings,
        "popular_games": popular_games,
    }
    with gzip.open(out_gz, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_gz) / (1024 * 1024)
    print(
        f"Wrote {out_gz} ({size_mb:.2f} MB) with {len(neighbors):,} games in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    bgg_csv = sys.argv[1] if len(sys.argv) > 1 else "data/BGG_Data.csv"
    ratings_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else r"E:\NUS SWS WebMining\Project\Test3.1\data\user_ratings.csv"
    )
    out_gz = sys.argv[3] if len(sys.argv) > 3 else "data/cf_topk.json.gz"
    build_cf_index(bgg_csv, ratings_csv, out_gz)
