# Codebase for CBMFW 4761 (Computational Genomics) course project: On the effectiveness of predicting transcription factor binding from chromatin accessibility and motif

Yiming Qu yq2355

## Computational Environment

Analysis is performed on Linux Ubuntu 20.04 with Python 3.10.6. Latest following packages and software are used:

- numpy
- scipy
- pandas
- scikit-learn
- pyBigWig
- igv_notebook
- BioPython
- matplotlib
- seaborn
- [UCSC Genome Browser and Blat application binaries](http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/)

## Data source

- [ENCODE](https://www.encodeproject.org/) - ATAC-seq and TF ChIP-seq
- [JASPAR](http://jaspar.genereg.net/) - Motif profiles
- [TFClass](http://tfclass.bioinf.med.uni-goettingen.de/) - Transcription factor classification

## Data aquisition and organization

### ENCODE metadata

Data tables crawled from ENCODE storing the metadata of *Experiments* and *Files* by running `./get_metadata.ipynb`. Each file corresponds to the metadata of a species and a data modality.

```
└── data_metadata
    ├── <species>_<data_modality>_experiments.csv.gz
    ├── <species>_<data_modality>_experiments.pkl
    ├── <species>_<data_modality>_files.csv.pkl
    └── ...
```

### ENCODE source data

Files downloaded directly from ENCODE by running `./scan/11-get_data.py`. Usage:
    
```shell
python3 11-get_data.py --species <species> --data_type <data_modality> --cell_type <cell_line> [--tf <TF_name>]
```
and follow the prompt.

Examples:

```shell
python3 11-get_data.py --species homo-sapiens --data_type atac-seq --cell_type K562
python3 11-get_data.py --species homo-sapiens --data_type tf-chip-seq --cell_type K562 --tf FOXA1
```

Files are organized by cell type and data modality.

```
└── data_encode
    ├── <cell_type>
    │   ├── ATAC-seq
    │   │   ├── <experiment_accession>_<replicate>_<file_accession>.bigWig
    │   │   ├── <experiment_accession>_<replicate>_<file_accession>.bed.gz
    │   │   └── <experiment_accession>_<replicate>_<file_accession>.bigBed
    │   ├── <TF>
    │   │   ├── <experiment_accession>_<replicate>_<file_accession>.bigWig
    │   │   ├── <experiment_accession>_<replicate>_<file_accession>.bed.gz
    │   │   └── <experiment_accession>_<replicate>_<file_accession>.bigBed
    │   └── ...
    └── ...
```

### JASPAR profiles

All profiles in the JASPAR core collection are donwloaded as instructed by [their GitHub](https://github.com/wassermanlab/JASPAR-UCSC-tracks) by running `get-profiles.py`.

Profiles are orgainzed by species.

```
└── data_jaspar_profile
    ├── <species>
    │   ├── <id>.<version>.jaspar
    │   ├── <id>.<version>.pwm
    │   └── ...
    ├── ...
    ├── get-profiles.py
    └── names.json
```

### JASPAR motif scanning

Results of motif scanning by JASPAR maintainers are downloaded in `.tsv.gz` format by running:

```shell
# define a list of TFs of interest and specify their JASPAR profiles
tfs=("NR3C1" "JUN" "REST" "CEBPB" "MAX" "FOXA1" "MYC" "CREB1" "FOSL1")
profiles=("MA0113.3" "MA0488.1" "MA0138.2" "MA0466.3" "MA0058.3" "MA0148.4" "MA0147.3" "MA0018.4" "MA0477.2")

for i in "${!tfs[@]}"; do
  tf="${tfs[$i]}"
  profile="${profiles[$i]}"
  wget "http://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022/hg38/$profile.tsv.gz"
done
```

and converted to `.bigBed` and `.bed.gz` with `./scan/util_scans2bigBed` (which uses `./scan/util_scans2bigbed_filter.py`):

```shell
for i in "${!tfs[@]}"; do
  tf="${tfs[$i]}"
  profile="${profiles[$i]}"
  ./scan/util_scans2bigBed -c <genome_size_file> -i <jaspar_scanning_dir>/scan/<species>/$profile.tsv.gz -o <jaspar_scanning_dir>/track/<species>/$profile.bb -t 4
done
```

Data are organized by species.

```
└── data_jaspar_scan
    ├── scan
    │   ├── <species>
    │   │   ├── <id>.<version>.tsv.gz
    │   │   └── ...
    │   └── ...
    ├── track
    │   ├── <species>
    │   │   ├── <id>.<version>.bb
    │   │   ├── <id>.<version>.bed.gz
    │   │   └── ...
    │   └── ...
    └── ...

```

### Genome-wide motif scanning

JASPAR motif scanning only reported position of motifs beyond a certain threshold. Genome-wide motif scanning is obtained by running `./scan/01-scan_sequence.py` with trivial threshold. Usage:

```shell
python3 01-scan_sequence.py <fasta_file> <profile_dir> --output_dir <data_motif_scan> --latest --taxon <taxon> --profile <profile_id> --pthresh 1 --rthresh 0
```

Example:
   
```shell
python3 01-scan_sequence.py data_genome/hg38.fa data_jaspar_profile --output_dir data_motif_scan --latest --taxon vertebrates --profile MA0113.3 --pthresh 1 --rthresh 0
```

The result is a very big `.tsv.gz` file whose number of rows is similar to the length of the genome. It is converted to two `.npz` files (corresponding to scan on positive and negative strand) with `./scan/02-process_scan.py`. Usage:

```shell
python3 02-process_scan.py --tf <TF_name> --assembly <assembly>
```

Results are organized by species and TF name
   
```
└── data_motif_scan
    ├── <species>
    │   ├── <TF_name>
    │   │   ├── <profile_id>.<profile_version>.-.npz
    │   │   ├── <profile_id>.<profile_version>.+.npz
    │   │   └── <profile_id>.<profile_version>.csv
    │   └── ...
    └── ...
```

The `.csv` file is a table with three columns storing the correspondence between motif score, p value and r value

`./examine_score.ipynb` can be used to visualize the relationship between these metrics (and to get Extended Fig 3.).

During scanning, metadata is stored in `./motif_metadata.json`.

### Overlap between ENCODE TF and JASPAR TF

ENCODE and JASPAR use separate naming system for TFs. `./process_matrix.ipynb` is used to get a subset of TFs that are shared without ambiguity between ENCODE and JASPAR, resulting in `./encode_chip_seq_matrix.json` and `./chosen_tfs.csv`.


## Data preprocessing and organization

### Sliding window aggregation results

`./scan/03-sample_window.py` is used to perform sliding Gaussian window aggregation at different window sizes and step sizes. Usage:

```shell
python3 03-sample_window.py --modality <modality> [--target <TF_name>] --cell_type <cell_type> --window_sizes <window_sizes> --step_sizes <step_sizes> [--temperature <temperature>] [--offset <offset>]
```

Example:

```shell
window_sizes="100 200 400 600 800"
step_sizes="10 20 40 60 80"

# for motif
python3 03-sample_window.py --modality motif --target NR3C1 --window_sizes $window_sizes --step_sizes $step_sizes --temperature 200 --offset 800

# for ATAC-seq
python3 scan/03-sample_window.py --modality epigenome --target ATAC-seq --cell_type NR3C1 --window_sizes $window_sizes --step_sizes $step_sizes

# for TF-ChIP-seq
python3 scan/03-sample_window.py --modality epigenome --target FOXA1 --cell_type HepG2 --window_sizes $window_sizes --step_sizes $step_sizes
```

## Data analysis and visualization

### Genome browser for a region on chromosome 6 in HepG2 for the TF FOXA1

`./genome_browser.ipynb` is used in Jupyter Notebook environment (not VS Code, not Jupyter Lab) to subset ATAC-seq data, FOXA1 ChIP-seq data and motif scanning data to the middle half of chromosome 6 in HepG2 cell line.

In this notebook, sliding window is also performed on this subset. Original signal and aggregated signal are converted to `.bigWig` format for visualization in genome browser. A visually appealing region featuring clear overlap between a ChIP-seq peak and ATAC-seq peak is identified by manual navigation. Track color and height are also subject to manual adjustment.

Genome browser presented in the manuscript is a screenshot of this region. 

### Quantitative peak-level analysis

Statistics shown in Fig 1b, Extended Fig 2, and Extended Fig 4 are obtained by `./data_problem.py`. Usage

```shell
python3 data_problem.py --cell_types <cell_lines> --tfs <tfs>
```

Apart from the analysis shown in the manuscript, this script also performs a dozen of other analysis. For example, an interesting observation is that while we expect independence between location of TF motif and ATAC-seq peaks, it seems that motifs are depleted in ATAC-seq peak regions. This is true for many cell line-TF pairs.

Resulting figures are saved in `./figs_data_problem`.

Figures are organized by cell line and TF name

```
└── figs_data_problem
    ├── <cell_line>
    │   ├── <TF_name>
    │   │   ├── 01-<analysis_description>.jpg
    │   │   ├── 02-<analysis_description>.jpg
    │   │   └── ...
    │   └── ...
    └── ...
```
 
### Random forest on window-level data

`./find_f.ipynb`, `./04-train_tf.py`

### Other files

`./data_downsample.ipynb`, `./find_f_old.ipynb`, `./get_data.ipynb` and `./scan/12-analysis_peak.py` were used in the early stage of the project.
