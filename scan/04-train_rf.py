"""Given a cell type and tf, fit a lightweight random forest for each chromosome (separately).
Train-test split is done by taking the last 20% windows on the chromosome (setting shuffle=False in the sklearn function train_test_split).
Save the model and spearman correlation.
"""

import os
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


def load_data(
    data_dir,
    cell_type,
    tf,
    window_size,
    step_size,
    temperature,
    offset,
    chr_,
    quantile_cutoff: float,
):
    atac_file = [
        f"{data_dir}/{cell_type}/ATAC-seq/{i}"
        for i in os.listdir(f"{data_dir}/{cell_type}/ATAC-seq")
        if i.startswith(f"{window_size}_{step_size}")
    ][0]
    chip_file = [
        f"{data_dir}/{cell_type}/{tf}/{i}"
        for i in os.listdir(f"{data_dir}/{cell_type}/{tf}")
        if i.startswith(f"{window_size}_{step_size}")
    ][0]
    motif_pos_file = f"{data_dir}/motif/hg38/{tf}/{window_size}_{step_size}_{temperature}_{offset}_pos.npz"
    motif_neg_file = f"{data_dir}/motif/hg38/{tf}/{window_size}_{step_size}_{temperature}_{offset}_neg.npz"

    atac = np.load(atac_file)
    chip = np.load(chip_file)
    motif_pos = np.load(motif_pos_file)
    motif_neg = np.load(motif_neg_file)

    atac_seq_data = atac[chr_]
    chip_seq_data = chip[chr_]
    motif_scan_pos_data = motif_pos[chr_]
    motif_scan_neg_data = motif_neg[chr_]

    data = pd.DataFrame(
        {
            "ATAC_seq": atac_seq_data,
            "ChIP_seq": chip_seq_data,
            "Motif_scan_pos": motif_scan_pos_data,
            "Motif_scan_neg": motif_scan_neg_data,
        }
    )
    data.dropna(inplace=True)

    data_high = data.loc[~(data <= data.quantile(quantile_cutoff)).all(axis=1)]
    return data_high


def fit_rf(data: pd.DataFrame, test_size: float, num_trees: int, seed: int) -> float:
    # Prepare the input and output data
    X = data[["ATAC_seq", "Motif_scan_pos", "Motif_scan_neg"]]
    y = data["ChIP_seq"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train the RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=num_trees, random_state=seed + 1)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)
    spearman_corr, p_value = spearmanr(y_test, y_pred)
    return rf, spearman_corr


if __name__ == "__main__":
    data_dir = "/home/ubuntu/project_comp/data_window"
    fig_dir = "./figs_rf"
    res_dir = "./results_rf"
    model_dir = "./models_rf"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str, required=True)
    parser.add_argument("--tf", type=str, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_trees", type=int, required=True)
    args = parser.parse_args()

    # Parameters
    cell_type = args.cell_type
    tf = args.tf
    window_size = args.window_size
    step_size = args.step_size
    temperature = args.temperature
    offset = args.offset
    num_trees = args.num_trees

    chrs = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
    ]

    spearman_dict = {}
    model_dict = {}
    for chr_ in tqdm(chrs):
        data = load_data(
            data_dir,
            cell_type,
            tf,
            window_size,
            step_size,
            temperature,
            offset,
            chr_,
            quantile_cutoff=0.9,
        )
        rf, spearman_corr = fit_rf(data, test_size=0.2, num_trees=num_trees, seed=42)
        spearman_dict[chr_] = spearman_corr
        model_dict[chr_] = rf

    with open(f"{res_dir}/sp_corr_rf_{num_trees}_{cell_type}_{tf}.json", "w") as f:
        json.dump(spearman_dict, f)

    with open(f"{model_dir}/rf_{num_trees}_{cell_type}_{tf}.pkl", "wb") as f:
        pickle.dump(model_dict, f)

    # plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(spearman_dict.keys(), spearman_dict.values(), color="darkorange")
    ax.set_xlabel("Chromosome", fontsize=16)
    ax.set_ylabel("Spearman correlation", fontsize=16)
    ax.set_title(
        f"ChIP-seq prediction performance with random forest ({num_trees} trees)",
        fontsize=18,
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    # x tick label rotation
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=30, horizontalalignment="right", fontsize=14
    )
    fig.savefig(
        f"{fig_dir}/sp_corr/sp_corr_rf_{num_trees}_{cell_type}_{tf}.jpg",
        dpi=150,
        bbox_inches="tight",
    )
