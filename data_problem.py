import argparse
import os
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pyBigWig
from rich.traceback import install

install()


def load_bed_file(data_dir: str) -> pd.DataFrame:
    file_path = [
        f"{data_dir}/{i}"
        for i in os.listdir(data_dir)
        if i.endswith(".bed") or i.endswith(".bed.gz")
    ][0]

    # 10 columns
    column_names = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "signalValue",
        "pValue",
        "qValue",
        "peak",
    ]
    dtype = {
        "chrom": str,
        "chromStart": int,
        "chromEnd": int,
        "name": str,
        "score": int,
        "strand": str,
        "signalValue": float,
        "pValue": float,
        "qValue": float,
        "peak": float,
    }
    try:
        data = pd.read_table(
            file_path, header=None, names=column_names, compression="gzip"
        ).sort_values(["chrom", "chromStart", "chromEnd"], ignore_index=True)
    except UnicodeDecodeError:
        data = pd.read_table(file_path, header=None, names=column_names).sort_values(
            ["chrom", "chromStart", "chromEnd"], ignore_index=True
        )

    data["peak_id"] = (
        data.chrom + ":" + data.chromStart.astype(str) + "-" + data.chromEnd.astype(str)
    )

    return data


def adjust_fontsize(ax, legend: bool = False) -> None:
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.title.set_size(20)
    if legend:
        ax.legend(fontsize=18)


def plot_peak_overlap(data: pd.DataFrame, fig_dir: str, fig_name: str, title: str):
    fig, ax = plt.subplots()
    data.groupby("peak_id").size().hist(ax=ax, bins=20)
    # set axis label
    ax.set_xlabel("Number of duplication")
    ax.set_ylabel("Number of peaks")
    ax.set_title(title)
    adjust_fontsize(ax)
    fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")


def plot_width_dist(
    data: pd.DataFrame, fig_dir: str, no_dup_only: bool, fig_name, title: str
):
    fig, ax = plt.subplots()
    peak_width = data.chromEnd - data.chromStart
    if no_dup_only:
        num_entries = data.groupby("peak_id")["peak_id"].transform("count")
        peak_width = peak_width[num_entries == 1]
    else:
        peak_width = peak_width.groupby(data.peak_id).first()
    mean, median, min_, max_ = (
        peak_width.mean(),
        peak_width.median(),
        peak_width.min(),
        peak_width.max(),
    )
    peak_width.hist(ax=ax, bins=50)
    # annotate the figure with mean, median, min, max at the top right corner
    ax.annotate(
        f"mean: {mean:.2f}\nmedian: {median:.2f}\nmin: {min_}\nmax: {max_}",
        xy=(0.6, 0.7),
        xycoords="axes fraction",
        fontsize=16,
        bbox=dict(boxstyle="round", fc="w", alpha=0.5),
    )
    ax.set_xlabel("Peak width")
    ax.set_ylabel("Number of peaks")
    ax.set_title(title)
    adjust_fontsize(ax)
    if no_dup_only:
        fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")
    else:
        fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")


def plot_peak_fraction(
    data: pd.DataFrame, genome_size: pd.DataFrame, fig_dir: str, fig_name: str, title
):
    peak_size = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        peak_cover = np.zeros(i.size)
        data_chr = data.query(f"chrom == '{i.chrom}'")
        for j in data_chr.itertuples():
            peak_cover[j.chromStart : j.chromEnd] += 1
        # print((peak_cover > 1).sum(), peak_cover.astype(bool).sum())
        peak_size[i.chrom] = peak_cover.astype(bool).sum()

    # plot for each chromosome fraction of the genome that are peaks
    frac = {}
    for i in genome_size.itertuples():
        # if peak_size[i.chrom] == 0:
        #     continue
        frac[i.chrom] = peak_size[i.chrom] / i.size
    chr_id = list(
        map(lambda x: int(x[3:]) if x[3:].isdigit() else ord(x[3:]), frac.keys())
    )
    fig, ax = plt.subplots(figsize=(20, 7))
    pd.Series(frac)[np.argsort(chr_id)].plot.bar(ax=ax)
    # set reasonable axis names and title, make axis and tick label font larger
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Fraction of the genome that are peaks")
    ax.set_title(title)
    adjust_fontsize(ax)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")


def plot_motif_fraction(
    motif_pos_bed: pd.DataFrame,
    motif_neg_bed: pd.DataFrame,
    genome_size: pd.DataFrame,
    fig_dir: str,
    fig_name: str,
    title: str,
):
    motif_size_pos = {i: 0 for i in genome_size.chrom}
    motif_size_neg = {i: 0 for i in genome_size.chrom}
    motif_size_either = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        motif_cover_pos = np.zeros(i.size)
        motif_cover_neg = np.zeros(i.size)
        motif_bed_pos_chr = motif_pos_bed.query(f"chrom == '{i.chrom}'")
        motif_bed_neg_chr = motif_neg_bed.query(f"chrom == '{i.chrom}'")
        for j in motif_bed_pos_chr.itertuples():
            motif_cover_pos[j.chromStart : j.chromEnd] += 1
        for j in motif_bed_neg_chr.itertuples():
            motif_cover_neg[j.chromStart : j.chromEnd] += 1
        motif_cover_either = motif_cover_pos + motif_cover_neg
        # print(
        #     (motif_cover_pos > 1).sum(),
        #     (motif_cover_neg > 1).sum(),
        #     (motif_cover_either > 1).sum(),
        #     motif_cover_pos.astype(bool).sum(),
        #     motif_cover_neg.astype(bool).sum(),
        #     motif_cover_either.astype(bool).sum(),
        # )

        motif_size_pos[i.chrom] = motif_cover_pos.astype(bool).sum()
        motif_size_neg[i.chrom] = motif_cover_neg.astype(bool).sum()
        motif_size_either[i.chrom] = motif_cover_either.astype(bool).sum()

    # plot for each chromosome fraction of the genome that are motifs (this time there will be 3 bars per chromosome)
    frac_pos = {}
    frac_neg = {}
    frac_either = {}
    for i in genome_size.itertuples():
        # if motif_size_either[i.chrom] == 0:
        #     continue
        frac_pos[i.chrom] = motif_size_pos[i.chrom] / i.size
        frac_neg[i.chrom] = motif_size_neg[i.chrom] / i.size
        frac_either[i.chrom] = motif_size_either[i.chrom] / i.size

    chr_id = list(
        map(lambda x: int(x[3:]) if x[3:].isdigit() else ord(x[3:]), frac_pos.keys())
    )

    fig, ax = plt.subplots(figsize=(20, 7))

    pd.DataFrame(
        {
            "Positive strand": pd.Series(frac_pos)[np.argsort(chr_id)],
            "Negative strand": pd.Series(frac_neg)[np.argsort(chr_id)],
            "Either strand": pd.Series(frac_either)[np.argsort(chr_id)],
        }
    ).plot.bar(ax=ax)

    # set reasonable axis names and title, make axis and tick label font larger
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Fraction of the genome that are motifs")
    ax.set_title(title)
    ax.tick_params(axis="both", which="major")
    adjust_fontsize(ax, legend=True)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")


def plot_motif_overlap(
    bed,
    motif_pos_bed,
    motif_neg_bed,
    motif_length: int,
    fig_dir,
    fig_name,
    title,
    sample_background: bool,
    binarize: bool,
    whole_motif: bool,
):
    dtype = bool if binarize else np.int16
    peak_motif = {i: 0 for i in genome_size.chrom}
    non_peak_motif = {i: 0 for i in genome_size.chrom}
    peak_size = {i: 0 for i in genome_size.chrom}
    non_peak_size = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        motif_cover_pos = np.zeros(i.size, dtype=dtype)
        motif_cover_neg = np.zeros(i.size, dtype=dtype)
        motif_bed_pos_chr = motif_pos_bed.query(f"chrom == '{i.chrom}'")
        motif_bed_neg_chr = motif_neg_bed.query(f"chrom == '{i.chrom}'")
        bed_chr = bed.query(f"chrom == '{i.chrom}'")

        if binarize:
            for j in motif_bed_pos_chr.itertuples():
                motif_cover_pos[j.chromStart : j.chromEnd] = True
            for j in motif_bed_neg_chr.itertuples():
                motif_cover_neg[j.chromStart : j.chromEnd] = True
            motif_cover_either = motif_cover_pos | motif_cover_neg
        else:
            if whole_motif:
                for j in motif_bed_pos_chr.itertuples():
                    motif_cover_pos[j.chromStart] += 1
                for j in motif_bed_neg_chr.itertuples():
                    motif_cover_neg[j.chromStart] += 1
            else:
                for j in motif_bed_pos_chr.itertuples():
                    motif_cover_pos[j.chromStart : j.chromStart + motif_length] += 1
                for j in motif_bed_neg_chr.itertuples():
                    motif_cover_neg[j.chromStart : j.chromStart + motif_length] += 1
            motif_cover_either = motif_cover_pos + motif_cover_neg

        if sample_background:
            if bed_chr.shape[0] == 0:
                continue
            bed_chr_bg = sample_background_regions(bed_chr, i.size, factor=2)

            if binarize:
                peak_motif[i.chrom] = bed_chr.apply(
                    lambda x: motif_cover_either[
                        x.chromStart + motif_length : x.chromEnd - motif_length
                    ].any(),
                    axis=1,
                ).sum()
                non_peak_motif[i.chrom] = bed_chr_bg.apply(
                    lambda x: motif_cover_either[
                        x.chromStart + motif_length : x.chromEnd - motif_length
                    ].any(),
                    axis=1,
                ).sum()
            else:
                offset = 0 if whole_motif else motif_length
                normalize = 1 if whole_motif else motif_length
                peak_motif[i.chrom] = bed_chr.apply(
                    lambda x: motif_cover_either[
                        x.chromStart + offset : x.chromEnd - motif_length
                    ].sum()
                    / normalize,
                    axis=1,
                ).sum()

                non_peak_motif[i.chrom] = bed_chr_bg.apply(
                    lambda x: motif_cover_either[
                        x.chromStart + offset : x.chromEnd - motif_length
                    ].sum()
                    / normalize,
                    axis=1,
                ).sum()

            peak_size[i.chrom] = bed_chr.shape[0]
            non_peak_size[i.chrom] = bed_chr_bg.shape[0]
        else:
            peak_cover = np.zeros(i.size, dtype=bool)
            for j in bed_chr.itertuples():
                peak_cover[j.chromStart : j.chromEnd] = True
            if binarize:
                peak_motif[i.chrom] = (peak_cover & motif_cover_either).sum()
                non_peak_motif[i.chrom] = (~peak_cover & motif_cover_either).sum()
            else:
                peak_motif[i.chrom] = motif_cover_either[peak_cover].sum()
                non_peak_motif[i.chrom] = motif_cover_either[~peak_cover].sum()
            peak_size[i.chrom] = peak_cover.sum()
            non_peak_size[i.chrom] = (~peak_cover).sum()

    frac_peak = {}
    frac_non_peak = {}
    for i in genome_size.itertuples():
        # if peak_motif[i.chrom] == 0:
        #     continue
        frac_peak[i.chrom] = (
            peak_motif[i.chrom] / peak_size[i.chrom] if peak_size[i.chrom] != 0 else 0
        )
        frac_non_peak[i.chrom] = (
            non_peak_motif[i.chrom] / non_peak_size[i.chrom]
            if non_peak_size[i.chrom] != 0
            else 0
        )

    chr_id = list(
        map(lambda x: int(x[3:]) if x[3:].isdigit() else ord(x[3:]), frac_peak.keys())
    )

    values_peak = []
    values_non_peak = []
    for k in frac_peak.keys():
        values_peak.append(frac_peak[k])
        values_non_peak.append(frac_non_peak[k])

    value_peak, value_non_peak = sum(values_peak), sum(values_non_peak)
    fc = max(value_peak, value_non_peak) / min(value_peak, value_non_peak)
    _, p = wilcoxon(values_peak, values_non_peak)

    fig, ax = plt.subplots(figsize=(20, 7))

    pd.DataFrame(
        {
            "Peak regions": pd.Series(frac_peak)[np.argsort(chr_id)],
            "Background": pd.Series(frac_non_peak)[np.argsort(chr_id)],
        }
    ).plot.bar(ax=ax)

    # annotate p value and fold change on top left of plot
    ax.annotate(
        f"p value = {p:.4f}\nFC = {fc:.2f}",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
        fontsize=16,
        bbox=dict(boxstyle="round", fc="w", alpha=0.5),
    )

    # set reasonable axis names and title, make axis and tick label font larger
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(
        f"Fraction of regions that {'have' if sample_background else 'are'} motifs"
    )
    ax.set_title(title)
    ax.tick_params(axis="both", which="major")
    adjust_fontsize(ax, legend=True)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.jpg", dpi=150, bbox_inches="tight")


def plot_chip_atac_overlap_heatmap(
    atac_bed, chip_bed, genome_size, fig_dir, fig_name, title
):
    non_peak = {i: 0 for i in genome_size.chrom}
    only_chip_peak = {i: 0 for i in genome_size.chrom}
    only_atac_peak = {i: 0 for i in genome_size.chrom}
    both_peak = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        atac_peak_cover = np.zeros(i.size, dtype=bool)
        chip_peak_cover = np.zeros(i.size, dtype=bool)
        atac_bed_chr = atac_bed.query(f"chrom == '{i.chrom}'")
        chip_bed_chr = chip_bed.query(f"chrom == '{i.chrom}'")
        for j in atac_bed_chr.itertuples():
            atac_peak_cover[j.chromStart : j.chromEnd] = True
        for j in chip_bed_chr.itertuples():
            chip_peak_cover[j.chromStart : j.chromEnd] = True
        non_peak[i.chrom] = (~atac_peak_cover & ~chip_peak_cover).sum()
        only_atac_peak[i.chrom] = (atac_peak_cover & ~chip_peak_cover).sum()
        only_chip_peak[i.chrom] = (chip_peak_cover & ~atac_peak_cover).sum()
        both_peak[i.chrom] = (atac_peak_cover & chip_peak_cover).sum()

    # frac_non_peak = {}
    # frac_only_atac_peak = {}
    # frac_only_chip_peak = {}
    # frac_both_peak = {}
    # for i in genome_size.itertuples():
    #     if only_atac_peak[i.chrom] == 0 or only_chip_peak[i.chrom] == 0:
    #         continue
    #     # chr_length = genome_size.query(f"chrom == '{i.chrom}'")["size"].iloc[0]
    #     frac_non_peak[i.chrom] = non_peak[i.chrom]
    #     frac_only_atac_peak[i.chrom] = only_atac_peak[i.chrom]
    #     frac_only_chip_peak[i.chrom] = only_chip_peak[i.chrom]
    #     frac_both_peak[i.chrom] = both_peak[i.chrom]

    # it looks ugly. So let's try 2x2 heatmap (ignore chromosomes, do it in one go for the entire genome).
    frac_non_peak = 0
    frac_only_atac_peak = 0
    frac_only_chip_peak = 0
    frac_both_peak = 0

    for i in genome_size.itertuples():
        # if only_atac_peak[i.chrom] == 0 or only_chip_peak[i.chrom] == 0:
        #     continue
        frac_non_peak += non_peak[i.chrom]
        frac_only_atac_peak += only_atac_peak[i.chrom]
        frac_only_chip_peak += only_chip_peak[i.chrom]
        frac_both_peak += both_peak[i.chrom]

    data_plot = np.array(
        [
            [frac_both_peak, frac_only_chip_peak],
            [frac_only_atac_peak, frac_non_peak],
        ]
    )
    data_plot = data_plot / data_plot.sum()

    fig, ax = plt.subplots()

    sns.heatmap(
        data_plot,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
        annot_kws={"size": 20},
    )
    ax.set_xlabel("ATAC-seq peak")
    ax.set_ylabel("ChIP-seq peak")
    ax.set_title(title)
    adjust_fontsize(ax)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.png", dpi=150, bbox_inches="tight")


def plot_chip_atac_overlap_bar(
    atac_bed, chip_bed, genome_size, fig_dir, fig_name, title
):
    # atac_peak = {i: 0 for i in genome_size.chrom}
    chip_peak = {i: 0 for i in genome_size.chrom}
    non_chip_peak = {i: 0 for i in genome_size.chrom}
    chip_peak_atac = {i: 0 for i in genome_size.chrom}
    non_chip_peak_atac = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        atac_peak_cover = np.zeros(i.size, dtype=bool)
        # chip_peak_cover = np.zeros(i.size, dtype=bool)
        atac_bed_chr = atac_bed.query(f"chrom == '{i.chrom}'")
        for j in atac_bed_chr.itertuples():
            atac_peak_cover[j.chromStart : j.chromEnd] = True
        # for j in chip_bed_dedup_chr.itertuples():
        #     chip_peak_cover[j.chromStart : j.chromEnd] = True
        # for j in atac_bed_dedup_chr.itertuples():
        #     if np.any(chip_peak_cover[j.chromStart : j.chromEnd]):
        #         atac_peak_chip[i.chrom] += 1
        chip_bed_chr = chip_bed.query(f"chrom == '{i.chrom}'")
        chip_bed_chr_bg = sample_background_regions(chip_bed_chr, i.size, factor=2)
        chip_peak[i.chrom] = chip_bed_chr.shape[0]
        non_chip_peak[i.chrom] = chip_bed_chr_bg.shape[0]
        for j in chip_bed_chr.itertuples():
            if np.any(atac_peak_cover[j.chromStart : j.chromEnd]):
                chip_peak_atac[i.chrom] += 1
        for j in chip_bed_chr_bg.itertuples():
            if np.any(atac_peak_cover[j.chromStart : j.chromEnd]):
                non_chip_peak_atac[i.chrom] += 1

    frac_peak = {}
    frac_non_peak = {}
    for i in genome_size.itertuples():
        # if chip_peak[i.chrom] == 0:
        #     continue
        frac_peak[i.chrom] = (
            chip_peak_atac[i.chrom] / chip_peak[i.chrom]
            if chip_peak[i.chrom] != 0
            else 0
        )
        frac_non_peak[i.chrom] = (
            non_chip_peak_atac[i.chrom] / non_chip_peak[i.chrom]
            if non_chip_peak[i.chrom] != 0
            else 0
        )

    chr_id = list(
        map(
            lambda x: int(x[3:]) if x[3:].isdigit() else ord(x[3:]),
            frac_peak.keys(),
        )
    )

    fig, ax = plt.subplots(figsize=(20, 9))
    pd.DataFrame(
        {
            "ChIP-seq peaks overlapping a ATAC-seq peak": pd.Series(frac_peak)[
                np.argsort(chr_id)
            ],
            "Background regions overlapping a ATAC-seq peak": pd.Series(frac_non_peak)[
                np.argsort(chr_id)
            ],
        }
    ).plot.bar(ax=ax)

    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Fraction of regions that overlap ATAC-seq peak")
    ax.set_title(title)
    ax.tick_params(axis="both", which="major")
    adjust_fontsize(ax, legend=True)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.png", dpi=150, bbox_inches="tight")


def plot_predict_chip_from_atac_motif(
    atac_bed,
    chip_bed,
    motif_pos_bed,
    motif_neg_bed,
    genome_size,
    motif_length,
    fig_dir,
    fig_name,
    title,
):
    atac_peak = {i: 0 for i in genome_size.chrom}
    atac_peak_chip_only = {i: 0 for i in genome_size.chrom}
    atac_peak_motif_only = {i: 0 for i in genome_size.chrom}
    atac_peak_both = {i: 0 for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        chip_peak_cover = np.zeros(i.size, dtype=bool)
        motif_pos_cover = np.zeros(i.size, dtype=bool)
        motif_neg_cover = np.zeros(i.size, dtype=bool)

        atac_bed_chr = atac_bed.query(f"chrom == '{i.chrom}'")
        chip_bed_chr = chip_bed.query(f"chrom == '{i.chrom}'")
        motif_pos_bed_chr = motif_pos_bed.query(f"chrom == '{i.chrom}'")
        motif_neg_bed_chr = motif_neg_bed.query(f"chrom == '{i.chrom}'")
        for j in chip_bed_chr.itertuples():
            chip_peak_cover[j.chromStart : j.chromEnd] = True
        for j in motif_pos_bed_chr.itertuples():
            motif_pos_cover[j.chromStart] = True
        for j in motif_neg_bed_chr.itertuples():
            motif_neg_cover[j.chromStart] = True
        motif_either_cover = np.logical_or(motif_pos_cover, motif_neg_cover)
        motif_only = 0
        chip_only = 0
        both = 0
        for j in atac_bed_chr.itertuples():
            if np.any(chip_peak_cover[j.chromStart : j.chromEnd]):
                if np.any(motif_either_cover[j.chromStart : j.chromEnd - motif_length]):
                    both += 1
                else:
                    chip_only += 1
            elif np.any(motif_either_cover[j.chromStart : j.chromEnd - motif_length]):
                motif_only += 1
        atac_peak[i.chrom] = atac_bed_chr.shape[0]
        atac_peak_chip_only[i.chrom] = chip_only
        atac_peak_motif_only[i.chrom] = motif_only
        atac_peak_both[i.chrom] = both

    frac_chip_only = 0
    frac_motif_only = 0
    frac_both = 0
    frac_neither = 0
    for i in genome_size.itertuples():
        # if atac_peak[i.chrom] == 0:
        #     continue
        frac_chip_only += atac_peak_chip_only[i.chrom]
        frac_motif_only += atac_peak_motif_only[i.chrom]
        frac_both += atac_peak_both[i.chrom]
        frac_neither += (
            atac_peak[i.chrom]
            - atac_peak_chip_only[i.chrom]
            - atac_peak_motif_only[i.chrom]
            - atac_peak_both[i.chrom]
        )

    data_plot = np.array(
        [
            [frac_both, frac_motif_only],
            [frac_chip_only, frac_neither],
        ]
    )

    data_plot = data_plot / data_plot.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots()
    sns.heatmap(
        data_plot,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
        annot_kws={"fontsize": 20},
    )
    ax.set_xlabel("ChIP-seq peak presence")
    ax.set_ylabel("Motif presence")
    ax.set_title(title)
    ax.tick_params(axis="both", which="major")
    adjust_fontsize(ax)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.png", dpi=150, bbox_inches="tight")


def plot_chip_peak_atac_signal(
    atac_track, chip_bed, genome_size, fig_dir, fig_name, title
):
    atac_track_meta = atac_track.chroms()
    atac_signal_peak = {i: [] for i in genome_size.chrom}
    atac_signal_bg = {i: [] for i in genome_size.chrom}
    for i in tqdm(genome_size.itertuples(), total=genome_size.shape[0]):
        if i.chrom not in atac_track_meta:
            continue
        chip_bed_chr = chip_bed.query(f"chrom == '{i.chrom}'")
        chip_bed_bg_chr = sample_background_regions(chip_bed_chr, i.size)
        atac_signal = atac_track.values(i.chrom, 0, i.size)
        for j in chip_bed_chr.itertuples():
            atac_signal_peak[i.chrom].append(
                np.mean(atac_signal[j.chromStart : j.chromEnd])
            )
        for j in chip_bed_bg_chr.itertuples():
            atac_signal_bg[i.chrom].append(
                np.mean(atac_signal[j.chromStart : j.chromEnd])
            )
    # plot two distributions
    atac_signal_peak_collect = np.array(
        [j for i in atac_signal_peak.values() for j in i]
    )
    atac_signal_bg_collect = np.array([j for i in atac_signal_bg.values() for j in i])
    fig, ax = plt.subplots()
    sns.kdeplot(
        {
            "ChIP-seq peak": atac_signal_peak_collect + 1,
            "Background": atac_signal_bg_collect + 1,
        },
        linewidth=3,
        label="Background",
        fill=True,
        ax=ax,
        cut=0,
        log_scale=True,
    )
    ax.set_xlabel("ATAC-seq signal")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.tick_params(axis="both", which="major")
    adjust_fontsize(ax)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{fig_name}.png", dpi=150, bbox_inches="tight")


def sample_background_regions(
    bed_chr: pd.DataFrame, chr_length: int, factor: int = 1, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starts = []
    ends = []
    for _ in range(factor):
        start = rng.integers(0, chr_length, size=bed_chr.shape[0])
        starts.append(start)
        end = start + (bed_chr.chromEnd - bed_chr.chromStart)
        ends.append(end)
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    return pd.DataFrame(dict(chromStart=starts, chromEnd=ends))


def plot_for_one_cell_type_tf(
    cell_type, tf, data_dir, fig_dir, motif_data_dir, genome_size, motif_meta
):
    atac_track_data_dir = f"./data/{cell_type}/ATAC-seq"

    profile = motif_meta[tf]["jaspar_id"]

    atac_bed = load_bed_file(f"{data_dir}/{cell_type}/ATAC-seq")
    chip_bed = load_bed_file(f"{data_dir}/{cell_type}/{tf}")
    atac_bed_dedup = atac_bed.drop_duplicates("peak_id")
    chip_bed_dedup = chip_bed.drop_duplicates("peak_id")
    motif_bed = pd.read_table(
        f"{motif_data_dir}/{profile}.bed.gz",
        header=None,
        names=["chrom", "chromStart", "chromEnd", "name", "score", "strand"],
    )
    motif_pos_bed, motif_neg_bed = (
        motif_bed[motif_bed.strand == "+"],
        motif_bed[motif_bed.strand == "-"],
    )
    motif_length = (motif_bed.chromEnd - motif_bed.chromStart).iloc[0]
    atac_track_data_path = [
        f"{atac_track_data_dir}/{i}"
        for i in os.listdir(atac_track_data_dir)
        if i.endswith(".bigWig")
    ][0]
    atac_track = pyBigWig.open(atac_track_data_path)

    # do the plots
    # ====
    plot_peak_overlap(
        atac_bed,
        fig_dir,
        fig_name="01-atac_overlap",
        title=f'The "duplication peak" problem of ATAC-seq peak in {cell_type}',
    )
    plot_width_dist(
        atac_bed,
        fig_dir,
        no_dup_only=False,
        fig_name="02-atac_width",
        title=f"Peak width distribution of ATAC-seq in {cell_type}",
    )
    plot_width_dist(
        atac_bed,
        fig_dir,
        no_dup_only=True,
        fig_name="03-atac_width_no_dup",
        title=f"Peak width distribution of ATAC-seq in {cell_type} (unique peaks)",
    )
    plot_peak_fraction(
        atac_bed_dedup,
        genome_size,
        fig_dir,
        fig_name="04-atac_fraction",
        title=f"Fraction of the genome that are ATAC-seq peak in {cell_type}",
    )

    # ====
    plot_peak_overlap(
        chip_bed,
        fig_dir,
        fig_name="05-chip_overlap",
        title=f'The "duplication peak" problem of {tf} ChIP-seq peak in {cell_type}',
    )
    plot_width_dist(
        chip_bed,
        fig_dir,
        no_dup_only=False,
        fig_name="06-chip_width",
        title=f"Peak width distribution of {tf} ChIP-seq peaks in {cell_type}",
    )
    plot_width_dist(
        chip_bed_dedup,
        fig_dir,
        no_dup_only=True,
        fig_name="07-chip_width_no_dup",
        title=f"Peak width distribution of {tf} ChIP-seq peaks in {cell_type} (unique peaks)",
    )
    plot_peak_fraction(
        chip_bed_dedup,
        genome_size,
        fig_dir,
        fig_name="08-chip_fraction",
        title=f"Fraction of the genome that are {tf} ChIP-seq peak in {cell_type}",
    )
    # ====
    plot_motif_fraction(
        motif_pos_bed,
        motif_neg_bed,
        genome_size,
        fig_dir,
        fig_name="09-motif_fraction",
        title=f"Fraction of the genome that are {tf} motif in hg38",
    )

    # ====
    overlap_title = f"Motif fraction across chromosomes for {tf} motif in {cell_type} in hg38 for ATAC-seq peak regions and background regions"
    plot_motif_overlap(
        atac_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="10-motif_overlap_atac_bin",
        title=overlap_title,
        binarize=True,
        sample_background=False,
        whole_motif=False,
    )
    plot_motif_overlap(
        atac_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="11-motif_overlap_atac_bin_sample",
        title=overlap_title,
        binarize=True,
        sample_background=True,
        whole_motif=False,
    )
    plot_motif_overlap(
        atac_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="12-motif_overlap_atac",
        title=overlap_title,
        binarize=False,
        sample_background=False,
        whole_motif=False,
    )
    plot_motif_overlap(
        atac_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="13-motif_overlap_atac_sample",
        title=overlap_title,
        binarize=False,
        sample_background=True,
        whole_motif=False,
    )
    plot_motif_overlap(
        atac_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="13-motif_overlap_atac_sample_whole",
        title=overlap_title,
        binarize=False,
        sample_background=True,
        whole_motif=True,
    )

    # ==== do the same thing for ChIP-seq peaks
    overlap_title = f"Motif fraction across chromosomes for {tf} motif in {cell_type} in hg38 for ChIP-seq peak regions and background regions"
    plot_motif_overlap(
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="14-motif_overlap_chip_bin",
        title=overlap_title,
        binarize=True,
        sample_background=False,
        whole_motif=False,
    )
    plot_motif_overlap(
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="15-motif_overlap_chip_bin_sample",
        title=overlap_title,
        binarize=True,
        sample_background=True,
        whole_motif=False,
    )
    plot_motif_overlap(
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="16-motif_overlap_chip",
        title=overlap_title,
        binarize=False,
        sample_background=False,
        whole_motif=False,
    )
    plot_motif_overlap(
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="17-motif_overlap_chip_sample",
        title=overlap_title,
        binarize=False,
        sample_background=True,
        whole_motif=False,
    )
    plot_motif_overlap(
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        motif_length,
        fig_dir,
        fig_name="18-motif_overlap_chip_sample_whole",
        title=overlap_title,
        binarize=False,
        sample_background=True,
        whole_motif=True,
    )
    # ====
    plot_chip_atac_overlap_heatmap(
        atac_bed_dedup,
        chip_bed_dedup,
        genome_size,
        fig_dir,
        fig_name="19-atac_chip_overlap_heatmap",
        title=f"Overlap between ATAC-seq and {tf} ChIP-seq peaks in {cell_type}",
    )
    plot_chip_atac_overlap_bar(
        atac_bed_dedup,
        chip_bed_dedup,
        genome_size,
        fig_dir,
        fig_name="20-atac_chip_overlap",
        title=f"ChIP-seq peaks that overlap ATAC-seq peaks for {tf} and {cell_type}",
    )
    # ====
    plot_predict_chip_from_atac_motif(
        atac_bed_dedup,
        chip_bed_dedup,
        motif_pos_bed,
        motif_neg_bed,
        genome_size,
        motif_length,
        fig_dir,
        fig_name="21-predict_chip_from_atac_motif",
        title=f"Effectiveness of predicting ChIP-seq peak from ATAC-seq peak with or without motif for {tf} and {cell_type}",
    )
    # ====
    plot_chip_peak_atac_signal(
        atac_track,
        chip_bed_dedup,
        genome_size,
        fig_dir,
        fig_name="22-chip_peak_atac_signal",
        title=f"Distribution of ATAC-seq signal in ChIP-seq peaks and background regions for {tf} and {cell_type}",
    )
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_types", nargs="+")
    parser.add_argument("--tfs", nargs="+")
    args = parser.parse_args()

    # parameters
    cell_types = args.cell_types
    tfs = args.tfs
    assert len(cell_types) == len(tfs)

    # data path
    data_dir = "./data"
    motif_data_dir = "./data_jaspar/track/hg38"
    motif_meta_path = "./motif_metadata.json"
    genome_size_path = "/home/ubuntu/s3/genomes/hg38/hg38.fa.sizes"

    # read data
    with open(motif_meta_path, "r") as f:
        motif_meta = json.load(f)
    genome_size = pd.read_table(genome_size_path, header=None, names=["chrom", "size"])
    # delete chrM
    genome_size = genome_size[genome_size["chrom"] != "chrM"]

    for cell_type, tf in zip(cell_types, tfs):
        print(f"Plotting {tf} in {cell_type}")
        fig_dir = f"./figs_data_problem/{cell_type}/{tf}"
        os.makedirs(f"./figs_data_problem/{cell_type}", exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        plot_for_one_cell_type_tf(
            cell_type,
            tf,
            data_dir,
            fig_dir,
            motif_data_dir,
            genome_size,
            motif_meta,
        )
