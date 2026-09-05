"""Packet-level features, decoded from parsed packet headers.

The statement requires both feature levels and names the packet-level ones
directly: TTL variance, TCP window size, fragment flags, payload distribution
and retransmission counts. None exist in the flow table, which is a summary --
by the time traffic is a flow record the header fields have been averaged away.

The obstacle was size. The packet tables are 272 GB across eighteen files, and
the raw-byte tables are 40 GB, of which the first 7.7 GB turned out to cover
only the benign Monday: the files are ordered by capture time, so downloading a
prefix buys a prefix of the week, and every attack day sits past the end of it.

Parquet is columnar, so the fix is to not download the files. Ten of roughly two
hundred and fifty columns carry everything named above, and a column chunk can
be fetched by range request on its own: 23 MB per file rather than 15 GB, and
0.4 GB rather than 272 GB for the set. What follows reads those columns
remotely and aggregates them per flow, which is the granularity the flow table
joins on and therefore the granularity the windows are built from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = "rdpahalavan/CIC-IDS2017"
N_FILES = 18
CACHE = Path(__file__).resolve().parents[2] / "data" / "packet_features.parquet"

# Only these are read. Everything else in the table -- Kerberos, SMB, DNS, the
# raw hex payloads -- stays on the server and is never transferred.
COLUMNS = ["flow_id", "protocol", "IP ttl", "IP frag", "IP flags", "IP len",
           "TCP window", "TCP flags", "TCP seq"]

PACKET_FEATURES = [
    "pkt_ttl_mean", "pkt_ttl_std", "pkt_ttl_min",
    "pkt_win_mean", "pkt_win_std", "pkt_win_zero_rate",
    "pkt_len_mean", "pkt_len_std",
    "pkt_df_rate", "pkt_mf_rate", "pkt_frag_rate",
    "pkt_retransmit_rate", "pkt_syn_rate", "pkt_rst_rate",
    "pkt_per_flow",
]


def _accumulate(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-flow sums for one batch, in a form that adds across files.

    Sums rather than means throughout: a flow's packets can straddle a file
    boundary, and sums combine by addition where means do not.
    """
    ttl = frame["IP ttl"].astype("float64")
    win = frame["TCP window"].astype("float64")
    ln = frame["IP len"].astype("float64")
    ipf = frame["IP flags"].fillna("")
    tcf = frame["TCP flags"].fillna("")

    out = pd.DataFrame({
        "flow_id": frame["flow_id"].astype("int64"),
        "n": 1.0,
        "ttl_s": ttl.fillna(0.0), "ttl_ss": (ttl ** 2).fillna(0.0),
        "ttl_n": ttl.notna().astype("float64"),
        "ttl_min": ttl,
        "win_s": win.fillna(0.0), "win_ss": (win ** 2).fillna(0.0),
        "win_n": win.notna().astype("float64"),
        "win_zero": (win == 0).astype("float64"),
        "len_s": ln.fillna(0.0), "len_ss": (ln ** 2).fillna(0.0),
        "len_n": ln.notna().astype("float64"),
        "df": ipf.str.contains("DF").astype("float64"),
        "mf": ipf.str.contains("MF").astype("float64"),
        "frag": (frame["IP frag"].fillna(0) > 0).astype("float64"),
        "syn": (tcf.str.contains("S") & ~tcf.str.contains("A")).astype("float64"),
        "rst": tcf.str.contains("R").astype("float64"),
    })
    grouped = out.groupby("flow_id", sort=False)
    agg = grouped.sum(numeric_only=True)
    agg["ttl_min"] = grouped["ttl_min"].min()

    # A retransmission is a sequence number seen more than once in a flow.
    # Counted inside the batch, so a repeat split across a file boundary is
    # missed; at eighteen boundaries against 87 million packets that is noise.
    seq = frame.loc[frame["TCP seq"].notna(), ["flow_id", "TCP seq"]]
    if len(seq):
        dup = seq.groupby(["flow_id", "TCP seq"], sort=False).size() - 1
        agg["retx"] = dup.groupby("flow_id").sum().reindex(agg.index).fillna(0.0)
    else:
        agg["retx"] = 0.0
    return agg


def _finalise(total: pd.DataFrame) -> pd.DataFrame:
    """Turn accumulated sums into the per-flow features."""
    def std(s, ss, n):
        n = n.replace(0, np.nan)
        var = (ss / n) - (s / n) ** 2
        return np.sqrt(var.clip(lower=0))

    n = total["n"]
    out = pd.DataFrame(index=total.index)
    out["pkt_ttl_mean"] = total["ttl_s"] / total["ttl_n"].replace(0, np.nan)
    out["pkt_ttl_std"] = std(total["ttl_s"], total["ttl_ss"], total["ttl_n"])
    out["pkt_ttl_min"] = total["ttl_min"]
    out["pkt_win_mean"] = total["win_s"] / total["win_n"].replace(0, np.nan)
    out["pkt_win_std"] = std(total["win_s"], total["win_ss"], total["win_n"])
    out["pkt_win_zero_rate"] = total["win_zero"] / total["win_n"].replace(0, np.nan)
    out["pkt_len_mean"] = total["len_s"] / total["len_n"].replace(0, np.nan)
    out["pkt_len_std"] = std(total["len_s"], total["len_ss"], total["len_n"])
    out["pkt_df_rate"] = total["df"] / n
    out["pkt_mf_rate"] = total["mf"] / n
    out["pkt_frag_rate"] = total["frag"] / n
    out["pkt_retransmit_rate"] = total["retx"] / n
    out["pkt_syn_rate"] = total["syn"] / n
    out["pkt_rst_rate"] = total["rst"] / n
    out["pkt_per_flow"] = n
    return out.fillna(0.0)


def build_packet_features(files: int = N_FILES, cache: Path = CACHE,
                          verbose: bool = True) -> pd.DataFrame:
    """Read the needed columns from the remote packet tables, per flow."""
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_url

    if cache.exists():
        return pd.read_parquet(cache)

    parts: list[pd.DataFrame] = []
    for i in range(1, files + 1):
        url = hf_hub_url(REPO, f"Packet-Fields/Packet_Fields_File_{i}.parquet",
                         repo_type="dataset")
        handle = pq.ParquetFile(fsspec.open(url).open())
        for batch in handle.iter_batches(batch_size=1_000_000, columns=COLUMNS):
            parts.append(_accumulate(batch.to_pandas()))
        if verbose:
            print(f"  file {i}/{files}: {sum(len(p) for p in parts):,} flow rows",
                  flush=True)

    total = pd.concat(parts).groupby(level=0).sum()
    # min() does not survive a sum; recover it by taking the min of the mins.
    total["ttl_min"] = pd.concat(
        [p["ttl_min"] for p in parts]).groupby(level=0).min()
    out = _finalise(total)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out
