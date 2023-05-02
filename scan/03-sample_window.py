"""
Given a window size (like 100) and a step size (like 10), aggregate the scores in 
motif scanning result, ChIP-seq signal, and ATAC-seq signal, with a gaussian kernel.
- ChIP-seq signal and ATAC-seq signal are signal p-value in bigwig format, directly downloaded from in internet
- Motif scanning is obtained locally, and stored in npz format. It is more lightweight. There is no need to use bigwig unless for visualization purpose.
"""

import json
import os
import argparse

import numpy as np
import pyBigWig
import joblib
from tqdm.auto import tqdm


def gaussian_kernel(window_size, sigma=1):
    x = np.arange(-(window_size - 1) / 2, (window_size - 1) / 2 + 1, 1)
    g = np.exp(-((x**2) / (2 * sigma**2)))
    return g / g.sum()


def get_start_center_end(len_chr, window_size, step_size):
    start = np.arange(0, len_chr - window_size + 1, step_size)
    end = start + window_size
    center = (start + end) // 2
    return start, center, end


def sample_window(
    signal_path,
    signal_type,
    window_size,
    step_size,
    n_jobs=8,
    temperature=200,
    offset=800,
) -> dict:
    # NOTE: could be nan if any of the values in the window is nan
    if signal_type == "motif":
        signal = np.load(signal_path, allow_pickle=True)
        chrs = list(signal.keys())
        chr2len = {chr_: len(signal[chr_]) for chr_ in chrs}

        def get_value(signal, start, window_size):
            window = signal[start : start + window_size]
            if (window == -10000).any():
                return np.full(window_size, np.nan)
            window = window - offset
            window = 1 / (1 + np.exp(-window / temperature))
            return window

    else:
        signal = pyBigWig.open(signal_path)
        chrs = list(signal.chroms().keys())
        chr2len = signal.chroms()

        def get_value(signal, start, window_size):
            return signal[start : start + window_size]

    def sample_window_one(signal, chr_):
        pos_start, _, _ = get_start_center_end(chr2len[chr_], window_size, step_size)
        scan = np.zeros(len(pos_start), dtype=np.float32)
        for idx, window_start in enumerate(tqdm(pos_start)):
            window_scan = get_value(signal, window_start, window_size)
            scan[idx] = (window_scan * kernel).sum().astype(np.float32)
        return scan

    kernel = gaussian_kernel(window_size, sigma=window_size // 6)
    if signal_type == "motif":
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(sample_window_one)(signal[chr_], chr_) for chr_ in chrs
        )
    else:
        # results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        #     sample_window_one(signal.values(chr_, 0, chr2len[chr_]), chr_)
        #     for chr_ in chrs
        # )
        results = []
        for chr_ in chrs:
            results.append(
                sample_window_one(signal.values(chr_, 0, chr2len[chr_]), chr_)
            )
    return dict(zip(chrs, results))

    # result = {}
    # for chr_ in signal:
    #     s = signal[chr_]
    #     len_chr = len(s)
    #     pos_start, _, _ = get_start_center_end(len_chr, window_size, step_size)
    #     scan = np.zeros(len(pos_start), dtype=np.float32)
    #     for idx, window_start in enumerate(tqdm(pos_start)):
    #         window_scan = get_value(s, window_start, window_size)
    #         scan[idx] = (window_scan * kernel).sum().astype(np.float32)
    #     result[chr_] = scan


def get_motif_signal_name(tf, motif_metadata_path, strand) -> str:
    with open(motif_metadata_path) as f:
        motif_metadata = json.load(f)
    meta = motif_metadata[tf]
    # return f"{meta['jaspar_id']}.{meta['version']}.{meta['length']}.{meta['min_score']}.{meta['max_score']}.{strand}.npz"
    return f"{meta['jaspar_id']}.{strand}.npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality", type=str, required=True, choices=["epigenome", "motif"]
    )
    parser.add_argument("--target", type=str)
    parser.add_argument("--cell_type", type=str)
    parser.add_argument("--window_sizes", type=int, nargs="+", default=[200, 400, 600])
    parser.add_argument("--step_sizes", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--temperature", type=int, default=200)
    parser.add_argument("--offset", type=int, default=800)

    args = parser.parse_args()
    # example: python3 03-sample_window.py --modality epigenome --target NR3C1 --cell_type HepG2
    # example: python3 03-sample_window.py --modality epigenome --target ATAC-seq --cell_type HepG2
    # example: python3 03-sample_window.py --modality motif --target NR3C1

    modality = args.modality
    target = args.target
    cell_type = args.cell_type
    window_sizes = args.window_sizes
    step_sizes = args.step_sizes
    temperature = args.temperature
    offset = args.offset

    assert len(window_sizes) == len(step_sizes)

    motif_metadata_path = "/home/ubuntu/project_comp/motif_metadata.json"
    save_dir = "data_window"

    assembly = "hg38"

    if modality == "motif":
        data_dir = "/home/ubuntu/s3/data/tf_motif_scan"
        os.makedirs(f"{save_dir}/motif/{assembly}/{target}", exist_ok=True)
        for window_size, step_size in zip(window_sizes, step_sizes):
            for s in ["+", "-"]:
                scan_name = get_motif_signal_name(target, motif_metadata_path, s)
                res = sample_window(
                    f"{data_dir}/{assembly}/{target}/{scan_name}",
                    "motif",
                    window_size,
                    step_size,
                    temperature=temperature,
                    offset=offset,
                )
                save_name = (
                    f"{window_size}_{step_size}_{temperature}_{offset}_pos.npz"
                    if s == "+"
                    else f"{window_size}_{step_size}_{temperature}_{offset}_neg.npz"
                )
                np.savez_compressed(
                    f"{save_dir}/motif/{assembly}/{target}/{save_name}", **res
                )
    elif modality == "epigenome":
        epigenome_dir = f"data/{cell_type}/{target}"
        epi_seq_paths = [i for i in os.listdir(epigenome_dir) if i.endswith(".bigWig")]
        os.makedirs(f"{save_dir}/{cell_type}/{target}", exist_ok=True)
        for window_size, step_size in zip(window_sizes, step_sizes):
            for chip_seq_path in epi_seq_paths:
                res = sample_window(
                    f"{epigenome_dir}/{chip_seq_path}",
                    modality,
                    window_size,
                    step_size,
                    n_jobs=1,
                )
                save_name = (
                    f"{window_size}_{step_size}_{chip_seq_path.split('.')[0]}.npz"
                )
                np.savez_compressed(
                    f"{save_dir}/{cell_type}/{target}/{save_name}", **res
                )

    # scan_dir = f"data_scan/{tf}_{assembly}"
    # epigenome_dir = f"data/{cell_type}"

    # scan_names = [get_motif_signal_name(tf, motif_metadata_path, s) for s in ["+", "-"]]

    # chip_seq_paths = [
    #     i for i in os.listdir(epigenome_dir) if i.startswith("tf-chip-seq")
    # ]
    # atac_seq_paths = [i for i in os.listdir(epigenome_dir) if i.startswith("atac-seq")]

    # for window_size, step_size in zip(window_sizes, step_sizes):
    #     for s in ["+", "-"]:
    #         scan_name = get_motif_signal_name(tf, motif_metadata_path, s)
    #         res = sample_window(
    #             f"{scan_dir}/{scan_name}", "motif", window_size, step_size
    #         )

    #         save_name = (
    #             f"{window_size}_{step_size}_motif-pos"
    #             if s == "+"
    #             else f"{window_size}_{step_size}_motif-neg"
    #         )
    #         np.savez_compressed(f"{save_dir}/{cell_type}_{tf}/{save_name}.npz", **res)
    #     for chip_seq_path in chip_seq_paths:
    #         res = sample_window(
    #             f"{epigenome_dir}/{chip_seq_path}",
    #             "tf-chip-seq",
    #             window_size,
    #             step_size,
    #             n_jobs=1,
    #         )
    #         save_name = f"{window_size}_{step_size}_{chip_seq_path.split('.')[0]}.npz"
    #         np.savez_compressed(f"{save_dir}/{cell_type}_{tf}/{save_name}", **res)
    #     for atac_seq_path in atac_seq_paths:
    #         res = sample_window(
    #             f"{epigenome_dir}/{atac_seq_path}",
    #             "atac-seq",
    #             window_size,
    #             step_size,
    #             n_jobs=1,
    #         )
    #         save_name = f"{window_size}_{step_size}_{atac_seq_path.split('.')[0]}.npz"
    #         np.savez_compressed(f"{save_dir}/{cell_type}_{tf}/{save_name}", **res)
