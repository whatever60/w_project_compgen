import argparse
import subprocess
import os

import numpy as np
import pandas as pd
from rich import print as rprint


# arguments: species, cell_type, tf, data_type
parser = argparse.ArgumentParser()

parser.add_argument(
    "--species",
    type=str,
    required=True,
    choices=[
        "homo-sapiens",
        "mus-musculus",
        "caenorhabditis-elegans",
        "trichechus-manatus",
        "drosophila-melanogaster",
        "drosophila-pseudoobscura",
    ],
)
parser.add_argument(
    "--data_type",
    type=str,
    required=True,
    choices=[
        "motif",
        "atac-seq",
        "tf-chip-seq",
        "control-chip-seq",
        "long-read-rna-seq",
        "polya-minus-rna-seq",
        "crispri-rna-seq",
        "snatac-seq",
        "total-rna-seq",
        "dnase-seq",
        "cage",
        "long-read-scrna-seq",
        "polya-plus-rna-seq",
        "scrna-seq",
        "in-situ-hi-c",
        "mnase-seq",
    ],
)
parser.add_argument("--cell_type", type=str, required=True)
parser.add_argument("--tf", type=str)

args = parser.parse_args()

species = args.species
cell_type = args.cell_type
tf = args.tf
data_type = args.data_type

species2assembly = {
    "homo-sapiens": "GRCh38",
    "mus-musculus": "mm10",
    "caenorhabditis-elegans": "ce11",
    "trichechus-manatus": "manNov1",
    "drosophila-melanogaster": "dm6",
    "drosophila-pseudoobscura": "dp6",
}
assembly = species2assembly[species]

if data_type == "motif":
    pass
else:
    # ==== load metadata ====
    rprint("Loading metadata")
    mother_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_file_path = f"{mother_dir}/data_metadata/{species}_{data_type}_files.pkl"
    metadata_exp_path = (
        f"{mother_dir}/data_metadata/{species}_{data_type}_experiments.pkl"
    )
    df_metadata_file = pd.read_pickle(metadata_file_path)
    df_metadata_exp = pd.read_pickle(metadata_exp_path)

    # ==== retrieve experiment metadata ====
    exp = df_metadata_exp.query(f"term_name == '{cell_type}' and perturbed == False")
    if data_type == "tf-chip-seq":
        exp = exp[exp.target.map(lambda x: x["label"]) == tf]

    # summarize the experiment metadata. specifically, I want to know how many experiments
    # there are for this cell type, how many biological_replicates they have, which lab they
    # come from.
    exp_ids = exp.accession.unique()
    if not len(exp_ids):
        rprint(f"No experiments found for {cell_type} {data_type} data.")
        exit(1)
    rprint(f"Found {len(exp)} experiments for {cell_type} {data_type} data: ")
    rprint("\t", end="")
    rprint(", ".join([f"[{i+1}] {c}" for i, c in enumerate(exp_ids)]))
    labs = exp.lab.map(lambda x: x["title"]).to_numpy()
    exp = exp.iloc[np.argsort(labs).tolist()]
    for lab in np.unique(labs):
        num_reps = exp[labs == lab].bio_replicate_count.astype(str).tolist()
        rprint(
            f"\tWithin these, {len(num_reps)} experiments come from {lab}, "
            f"with {', '.join(num_reps)} biological replicates (respectively)."
        )

    user_input = input("\tEnter the experiment to download: ")

    # validate the user input and get the selected replicate configuration
    if not user_input.isdigit() or int(user_input) not in range(1, len(exp_ids) + 1):
        rprint("Invalid input.")
        exit(1)

    accession_exp = exp_ids[int(user_input) - 1]

    # ==== retrieve file metadata ====
    file_ = df_metadata_file.query(
        f"experiment_id == '/experiments/{accession_exp}/' and assembly_x == '{assembly}'"
    )

    if data_type == "tf-chip-seq":
        output_types = ["signal p-value", "IDR thresholded peaks"]
        file_format = ["bigWig", "bigBed", "bed"]
    elif data_type == "atac-seq":
        output_types = ["signal p-value", "pseudoreplicated peaks"]
        file_format = ["bigWig", "bigBed", "bed"]
    else:
        raise NotImplementedError

    file_ = file_[
        file_.output_type.isin(output_types) & file_.file_format.isin(file_format)
    ]

    # ask the user to select a replicate configuration
    replicate_configs = file_.biological_replicates.apply(
        lambda x: "".join(map(str, x))
    ).unique()
    replicate_configs_str = ", ".join(
        [f"[{i+1}] {c}" for i, c in enumerate(replicate_configs)]
    )

    rprint(
        f"Found {len(replicate_configs)} replicate configurations: {replicate_configs_str}"
    )

    user_input_rep = input("\tEnter the replicate configuration to download: ")

    # validate the user input and get the selected replicate configuration
    if not user_input_rep.isdigit() or int(user_input_rep) not in range(
        1, len(replicate_configs) + 1
    ):
        rprint("Invalid input.")
        exit(1)

    replicate_config = replicate_configs[int(user_input_rep) - 1]

    file_ = file_[
        file_.biological_replicates.apply(lambda x: "".join(map(str, x)))
        == replicate_config
    ]

    file_.accession = file_.file_id.map(lambda x: x.split("/")[-2])

    # ==== download files ====

    # for each output type, only keep the file with the latest release date
    for format in file_format:
        file_of_type = file_[file_.file_format == format]
        if len(file_of_type) > 1:
            file_of_type = file_of_type.sort_values("cloud_metadata")
            file_wanted = file_of_type.iloc[-1]
            rprint(f"Found {len(file_of_type)} files of type {format}.")
            rprint(
                f"\tKeeping the file with the latest release date ({file_wanted.accession}) automatically."
            )
            file_.drop(file_of_type.iloc[:-1].index, inplace=True, axis=0)

    if data_type == "tf-chip-seq":
        dir_name = tf
    elif data_type == "atac-seq":
        dir_name = "ATAC-seq"
    else:
        raise NotImplementedError

    os.makedirs(f"{mother_dir}/data/{cell_type}", exist_ok=True)
    os.makedirs(f"{mother_dir}/data/{cell_type}/{dir_name}", exist_ok=True)

    for c in file_.itertuples():
        # subprocess.run(
        #     [
        #         "wget",
        #         "-O",
        #         "-q",
        #         "--show-progress",
        #         f"{mother_dir}/data/{cell_type}/{dir_name}/{accession_exp}_rep{replicate_config}_{c.accession}.{c.file_format}",
        #         c.cloud_metadata,
        #     ]
        # )
        s3_path = f"s3://encode-public/{'/'.join(c.cloud_metadata.split('/')[3:-1])}"
        file_name = c.cloud_metadata.split("/")[-1]
        download_dir = f"{mother_dir}/data/{cell_type}/{dir_name}"
        subprocess.run(["aws", "s3", "sync", s3_path, download_dir])
        subprocess.run(
            [
                "mv",
                f"{download_dir}/{file_name}",
                f"{download_dir}/{accession_exp}_rep{replicate_config}_{file_name}",
            ]
        )
        rprint("Downloaded")
