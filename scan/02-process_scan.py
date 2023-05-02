"""
Process tsv output by `matrix_scan` to two bigWig files, one for positive strand and one for negative strand. 
"""

import argparse
import math
import gzip
import os
import json

import numpy as np
import pandas as pd
import pyBigWig
from tqdm.auto import tqdm


def get_motif_length(scan_result_path: str):
    """get length of motif"""
    try:
        (
            profile,
            version,
            length,
            min_score,
            min_score,
            tsv,
            gz,
        ) = scan_result_path.split(".")
        length_motif = int(length)
    except ValueError:
        with gzip.open(scan_result_path, "rb") as f:
            _, start, end, _, _ = next(f).decode("utf-8").strip().split("\t")
            length_motif = int(end) - int(start)
    return length_motif


def scan2bw(
    scan_result_path: str,
    chr_size_path_hg38: str,
    output_dir: str,
    profile_name: str,
    s: str,
    use_pandas=True,
):
    # using pandas is much faster, less memory intensive and more elegant.
    df_chr_size = pd.read_csv(chr_size_path_hg38, sep="\t", header=None)
    size_hg38 = df_chr_size.iloc[:, 1].sum()

    motif_length = get_motif_length(scan_result_path)

    batch_size = 10_000_000

    profile, version, length, min_score, max_score, tsv, gz = os.path.basename(
        scan_result_path
    ).split(".")
    output_path = (
        f"{output_dir}/{profile}.{version}.{length}.{min_score}.{max_score}.{s}.bw"
    )

    if os.path.isfile(output_path):
        # remove existing file
        os.remove(output_path)

    bw = pyBigWig.open(output_path, "w")
    bw.addHeader(list(df_chr_size.itertuples(index=False)), maxZooms=0)

    if use_pandas:
        for chunk in tqdm(
            pd.read_csv(scan_result_path, sep="\t", header=None, chunksize=batch_size),
            total=math.ceil(size_hg38 * 2 / batch_size),
        ):
            chunk.columns = ["chr", "start", "end", "score", "strand"]
            chunk = chunk.query("strand == @s")

            chrs = chunk["chr"].to_numpy().astype("U5")
            starts = chunk["start"].to_numpy().astype(np.int64)
            scores = chunk["score"].to_numpy().astype(np.float64)

            shift = motif_length // 2
            starts += shift

            chrs = chunk.chr.unique()
            for chr_ in chrs:
                chr_mask = chunk.chr == chr_
                bw.addEntries(
                    chr_,
                    starts[chr_mask],
                    values=scores[chr_mask],
                    span=1,
                )
    else:
        chr_str2int = {chr_: i for i, chr_ in enumerate(df_chr_size.iloc[:, 0])}
        chr_int2str = {i: chr_ for i, chr_ in enumerate(df_chr_size.iloc[:, 0])}
        chrs = np.full(size_hg38, -1, dtype=np.int8)  # chr
        starts = np.full(size_hg38, -1, dtype=np.int32)  # 0-based int
        scores = np.full(size_hg38, -10000, dtype=np.int16)
        with gzip.open(scan_result_path, "rb") as f:
            # \t separated file with 5 columns: chr, start, end, score, strand. 0-based
            length = 0
            for line in tqdm(f, total=size_hg38 * 2):
                chr_, start, _, score, strand = line.decode("utf-8").strip().split("\t")
                if strand == s:
                    chrs[length] = chr_str2int[chr_]
                    starts[length] = int(start)
                    scores[length] = int(score)
                    length += 1

        print(f"length of {s} strand: {length}")

        shift = motif_length // 2
        starts += shift
        ends = starts + 1

        chrs_strand = chrs[0, :length]
        starts_strand = starts[0, :length]
        ends_strand = ends[0, :length]
        scores_strand = scores[0, :length]
        for i in range(0, length, batch_size):
            bw.addEntries(
                map(lambda x: chr_int2str[x], chrs_strand[i : i + batch_size]),
                starts_strand[i : i + batch_size],
                ends=ends_strand[i : i + batch_size],
                values=scores_strand[i : i + batch_size],
            )
    bw.close()


def scan2npz(
    scan_result_path: str,
    scan_meta_path: str,
    chr_size_path_hg38: str,
    output_dir: str,
    s: str,
):
    """save a dict {chr: np.array} to npz file. Each array is a 1D array of scores,
    with length equal to the length of the chromosome.
    type is int16, with -10000 as missing value.
    """
    df_chr_size = pd.read_csv(chr_size_path_hg38, sep="\t", header=None)
    size_hg38 = df_chr_size.iloc[:, 1].sum()

    df_meta = pd.read_csv(scan_meta_path)
    min_score, max_score = df_meta.score.min(), df_meta.score.max()

    length = get_motif_length(scan_result_path)

    profile, version, tsv, gz = os.path.basename(scan_result_path).split(".")

    length = int(length)
    shift = length // 2

    batch_size = 10_000_000

    res_dict = {
        chr_: np.full(length, -10000, dtype=np.int16)
        for chr_, length in df_chr_size.itertuples(index=False)
    }
    for chunk in tqdm(
        pd.read_csv(scan_result_path, sep="\t", header=None, chunksize=batch_size),
        total=math.ceil(size_hg38 * 2 / batch_size),
    ):
        chunk.columns = ["chr", "start", "end", "score", "strand"]
        chunk = chunk.query("strand == @s")

        chrs = chunk["chr"].to_numpy()
        starts = chunk["start"]
        scores = chunk["score"].to_numpy().astype(np.int16)

        starts += shift

        chrs = chunk.chr.unique()
        for chr_ in chrs:
            chr_mask = chunk.chr == chr_
            res_dict[chr_][starts[chr_mask]] = scores[chr_mask]

    # output_path = (
    #     f"{output_dir}/{profile}.{version}.{length}.{min_score}.{max_score}.{s}.npz"
    # )

    output_path = f"{output_dir}/{profile}.{version}.{s}.npz"

    if os.path.isfile(output_path):
        # remove existing file
        os.remove(output_path)

    np.savez_compressed(output_path, **res_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str)
    parser.add_argument("--assembly", type=str)

    args = parser.parse_args()
    tf = args.tf
    assembly = args.assembly

    chr_size_path = f"/home/ubuntu/s3/genomes/{assembly}/{assembly}.fa.sizes"
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = f"{work_dir}/data_scan/{assembly}/{tf}"
    motif_meta_path = f"{work_dir}/motif_metadata.json"

    with open(motif_meta_path, "r") as f:
        motif_meta = json.load(f)
        profile = motif_meta[tf]["jaspar_id"]

    scan_result_path = f"{output_dir}/{profile}.tsv.gz"
    scan_meta_path = f"{output_dir}/{profile}.csv"

    # scan2bw(
    #     scan_result_path,
    #     chr_size_path_hg38,
    #     output_dir,
    #     profile,
    #     "+",
    # )

    # scan2bw(
    #     scan_result_path,
    #     chr_size_path_hg38,
    #     output_dir,
    #     profile,
    #     "-",
    # )

    scan2npz(
        scan_result_path,
        scan_meta_path,
        chr_size_path,
        output_dir,
        "-",
    )

    scan2npz(
        scan_result_path,
        scan_meta_path,
        chr_size_path,
        output_dir,
        "+",
    )
