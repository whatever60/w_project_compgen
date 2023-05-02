"""
Plots to make (given a cell type and a TF and a chromosome):
1. A barplot comparing the number of motifs and ATAC-seq peaks and ChIP-seq peaks.
2. A 2x2 heatmap showing the overlap between ATAC peaks and ChIP peaks, conditioned on the presence of the motif.
3. A 2x2 heatmap showing the overlap between ChIP peaks and ATAC peaks, conditioned on the presence of the motif.
4. A histogram showing the distribution of widths of the peaks.
"""

import os
import json
import argparse
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_venn
import pyBigWig
from tqdm.auto import tqdm

work_dir = os.path.dirname(os.path.dirname(__file__))

args = argparse.ArgumentParser()

args.add_argument("--cell_type", type=str, required=True)
args.add_argument("--tf", type=str, required=True)

args = args.parse_args()

cell_type = args.cell_type
tf = args.tf


def load_beds(cell_type, tf):
    encode_data_dir = f"{work_dir}/data/{cell_type}"
    with open(f"{work_dir}/motif_metadata.json", "r") as f:
        motif_metadata = json.load(f)
        profile = (
            motif_metadata[tf]["jaspar_id"] + "." + str(motif_metadata[tf]["version"])
        )

    # bigBedToBed
    bigbed_atac_path = list(
        filter(
            lambda x: x.endswith(".bigBed"),
            os.listdir(f"{encode_data_dir}/ATAC-seq"),
        )
    )[0]
    bigbed_chip_path = list(
        filter(lambda x: x.endswith(".bigBed"), os.listdir(f"{encode_data_dir}/{tf}"))
    )[0]

    bigbed_atac_path = f"{encode_data_dir}/ATAC-seq/{bigbed_atac_path}"
    bigbed_chip_path = f"{encode_data_dir}/{tf}/{bigbed_chip_path}"
    bed_atac_path = bigbed_atac_path.replace(".bigBed", ".bed")
    bed_chip_path = bigbed_chip_path.replace(".bigBed", ".bed")

    if not os.path.exists(f"{encode_data_dir}/{tf}/motif_peak"):
        subprocess.run(["bigBedToBed", bigbed_chip_path, bed_chip_path])

    if not os.path.exists(f"{encode_data_dir}/ATAC-seq/motif_peak"):
        subprocess.run(["bigBedToBed", bigbed_atac_path, bed_atac_path])

    motif_peak_path = f"{work_dir}/data_jaspar/track/hg38/{profile}.bed"

    # ==== read data ====
    motif_peak_df = pd.read_table(motif_peak_path, header=None)
    motif_peak_df.columns = ["chr", "start", "end", "name", "score", "strand"]
    motif_pos_peak_df = motif_peak_df[motif_peak_df["strand"] == "+"]
    motif_neg_peak_df = motif_peak_df[motif_peak_df["strand"] == "-"]
    chip_peak_df = pd.read_table(bed_chip_path, header=None)
    chip_peak_df.columns = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",
        "pValue",
        "qValue",
        "peak",
    ]
    atac_peak_df = pd.read_table(bed_atac_path, header=None)
    atac_peak_df.columns = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",
        "pValue",
        "qValue",
        "peak",
    ]

    return motif_pos_peak_df, motif_neg_peak_df, chip_peak_df, atac_peak_df


def plot_peak_count(
    cell_type,
    tf,
    motif_pos_peak_df,
    motif_neg_peak_df,
    chip_peak_df,
    atac_peak_df,
):
    """A line plot where the x axis is chromosome name and the y axis is the number of peaks.
    There will be 4 lines.
    """
    # ==== plot peak count ====
    motif_pos_peak_groupby_chr = motif_pos_peak_df.groupby("chr")
    motif_neg_peak_groupby_chr = motif_neg_peak_df.groupby("chr")
    chip_peak_groupby_chr = chip_peak_df.groupby("chr")
    atac_peak_groupby_chr = atac_peak_df.groupby("chr")

    # Get unique chromosomes from each dataframe
    chr_motif_pos = set(motif_pos_peak_df["chr"].unique())
    chr_motif_neg = set(motif_neg_peak_df["chr"].unique())
    chr_chip = set(chip_peak_df["chr"].unique())
    chr_atac = set(atac_peak_df["chr"].unique())

    # Find common chromosomes among all dataframes
    shared_chrs = chr_motif_pos & chr_motif_neg & chr_chip & chr_atac

    chr2peak_count = {}

    for chr in shared_chrs:
        chr2peak_count[chr] = {
            "motif_pos": len(motif_pos_peak_groupby_chr.get_group(chr)),
            "motif_neg": len(motif_neg_peak_groupby_chr.get_group(chr)),
            "chip": len(chip_peak_groupby_chr.get_group(chr)),
            "atac": len(atac_peak_groupby_chr.get_group(chr)),
        }

    chr2peak_count_df = pd.DataFrame.from_dict(chr2peak_count, orient="index")
    chr2peak_count_df = chr2peak_count_df.sort_index()
    sorted_chrs = sorted(
        shared_chrs, key=lambda x: int(x[3:]) if x[3:].isdigit() else 999
    )
    chr2peak_count_df = chr2peak_count_df.reindex(sorted_chrs)

    fig, ax = plt.subplots(figsize=(10, 5))

    # sns.lineplot(
    #     data=chr2peak_count_df,
    #     ax=ax,
    #     dashes=False,
    # )

    # set a color palette
    cm = plt.get_cmap("Set2")
    
    ax.plot(
        chr2peak_count_df.index,
        chr2peak_count_df["motif_pos"],
        marker="o",
        linestyle="dashed",
        label="Motif on positive strand",
        color=cm(0),
    )
    ax.plot(
        chr2peak_count_df.index,
        chr2peak_count_df["motif_neg"],
        marker="o",
        linestyle="dashed",
        label="Motif on negative strand",
        color=cm(1),
    )
    ax.plot(
        chr2peak_count_df.index,
        chr2peak_count_df["chip"],
        marker="o",
        linestyle="dashed",
        label="ChIP-seq peaks",
        color=cm(2),
    )
    ax.plot(
        chr2peak_count_df.index,
        chr2peak_count_df["atac"],
        marker="o",
        linestyle="dashed",
        label="ATAC-seq peaks",
        color=cm(3),
    )

    ax.set_title(f"{tf}, {cell_type}", fontsize=16)
    ax.set_yscale("log")
    ax.set_xlabel("Chromosome", fontsize=14)
    ax.set_ylabel("Peak count", fontsize=14)
    # rotate xtick labels by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # ax.legend(
    #     [
    #         "Motif on positive strand",
    #         "Motif on negative strand",
    #         "ChIP-seq peaks",
    #         "ATAC-seq peaks",
    #     ],
    # )
    # put the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # bigger tick labels
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(
        f"{work_dir}/figs/peak_count/{cell_type}_{tf}.png",
        dpi=150,
        bbox_inches="tight",
    )


# Precompute indices for each chromosome
def precompute_chr_indices(df):
    unique_chrs = df["chr"].unique()
    chr_indices = {}
    for chr_value in unique_chrs:
        left = np.searchsorted(df["chr"], chr_value, side="left")
        right = np.searchsorted(df["chr"], chr_value, side="right")
        chr_indices[chr_value] = (left, right)
    return chr_indices


if __name__ == "__main__":
    motif_pos_peak_df, motif_neg_peak_df, chip_peak_df, atac_peak_df = load_beds(
        cell_type, tf
    )

    plot_peak_count(
        cell_type,
        tf,
        motif_pos_peak_df,
        motif_neg_peak_df,
        chip_peak_df,
        atac_peak_df,
    )


# # Create an IntervalIndex for motif_peak_df and chip_peak_df
# motif_intervals = pd.IntervalIndex.from_arrays(
#     motif_peak_df["start"], motif_peak_df["end"], closed="both"
# )
# chip_intervals = pd.IntervalIndex.from_arrays(
#     chip_peak_df["start"], chip_peak_df["end"], closed="both"
# )

# motif_chr_indices = precompute_chr_indices(motif_peak_df)
# chip_chr_indices = precompute_chr_indices(chip_peak_df)

# # Initialize the output arrays
# motif_in_atac_peak = np.zeros(len(atac_peak_df), dtype=bool)
# atac_peak_overlap_chip_peak = np.zeros(len(atac_peak_df), dtype=bool)
# motif_length = motif_peak_df["end"].iloc[0] - motif_peak_df["start"].iloc[0]

# # Iterate over atac_peak_df
# for i, row in tqdm(atac_peak_df.iterrows(), total=len(atac_peak_df)):
#     atac_interval = pd.Interval(row["start"], row["end"], closed="both")
#     shrunk_atac_interval = pd.Interval(
#         row["start"] + motif_length, row["end"] - motif_length, closed="both"
#     )

#     # Get the precomputed indices for the current "chr" value in motif_peak_df and chip_peak_df
#     motif_chr_idx = motif_chr_indices.get(row["chr"], (0, 0))
#     chip_chr_idx = chip_chr_indices.get(row["chr"], (0, 0))

#     # Check if there's any overlap between the ATAC-seq peak and motif_peak_df
#     # motif_in_atac_peak[i] = any(
#     #     motif_intervals[motif_chr_idx[0] : motif_chr_idx[1]].overlaps(atac_interval)
#     # )
#     motif_in_atac_peak[i] = any(
#         motif_intervals[motif_chr_idx[0] : motif_chr_idx[1]].overlaps(
#             shrunk_atac_interval
#         )
#     )

#     # Check if there's any overlap between the ATAC-seq peak and chip_peak_df
#     atac_peak_overlap_chip_peak[i] = any(
#         chip_intervals[chip_chr_idx[0] : chip_chr_idx[1]].overlaps(atac_interval)
#     )

# # Add the results as new columns to atac_peak_df
# atac_peak_df["peak_has_motif"] = motif_in_atac_peak
# atac_peak_df["overlap_chip_seq_peak"] = atac_peak_overlap_chip_peak
