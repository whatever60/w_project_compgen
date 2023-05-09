#!/bin/bash

window_sizes="100 200 400 600 800"
step_sizes="10 20 40 60 80"
profiles=("MA0113.3" "MA0488.1" "MA0138.2" "MA0466.3" "MA0058.3" "MA0148.4" "MA0147.3" "MA0018.4" "MA0477.2")
tfs=("NR3C1" "JUN" "REST" "CEBPB" "MAX" "FOXA1" "MYC" "CREB1" "FOSL1")

for i in "${!tfs[@]}"; do
  tf="${tfs[$i]}"
  profile="${profiles[$i]}"
  wget "http://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022/hg38/$profile.tsv.gz"
done

for i in "${!tfs[@]}"; do
  tf="${tfs[$i]}"
  profile="${profiles[$i]}"
  ./scan/util_scans2bigBed -c /home/ubuntu/s3/genomes/hg38/hg38.fa.sizes -i /home/ubuntu/project_comp/data_jaspar/scan/hg38/$profile.tsv.gz -o /home/ubuntu/project_comp/data_jaspar/track/hg38/$profile.bb -t 4
done

for i in "${!tfs[@]}"; do
 tf="${tfs[$i]}"
 profile="${profiles[$i]}"
 echo $tf
 python3 scan/03-sample_window.py --modality motif --target "$tf" --window_sizes $window_sizes --step_sizes $step_sizes --temperature 200 --offset 800
done

cell_lines=("A549" "HepG2" "K562" "GM12878" "MCF-7" "WTC11" "SK-N-SH" "HCT116" "IMR-90")

for cell_line in "${cell_lines[@]}"; do
  echo $cell_line
  python3 scan/03-sample_window.py --modality epigenome --target ATAC-seq --cell_type "$cell_line" --window_sizes $window_sizes --step_sizes $step_sizes
done

cell_lines=("A549" "A549" "A549" "A549" "A549" "HepG2" "HepG2" "HepG2" "HepG2" "HepG2" "HepG2" "HepG2" "K562" "K562" "K562" "K562" "K562" "K562" "K562" "K562" "K562" "GM12878" "GM12878" "GM12878" "GM12878" "GM12878" "MCF-7" "MCF-7" "MCF-7" "MCF-7" "MCF-7" "MCF-7" "WTC11" "WTC11" "WTC11" "WTC11" "H1" "H1" "H1" "H1" "H1" "H1" "SK-N-SH" "SK-N-SH" "HCT116" "HCT116" "HCT116" "HCT116" "IMR-90")
tfs=("NR3C1" "JUN" "REST" "CEBPB" "MAX" "FOXA1" "NR3C1" "JUN" "MAX" "MYC" "CREB1" "FOSL1" "FOXA1" "REST" "CEBPB" "MAX" "MYC" "FOSL1" "CREB1" "NR3C1" "JUN" "REST" "CEBPB" "MAX" "MYC" "CREB1" "FOXA1" "REST" "CEBPB" "MAX" "CREB1" "JUN" "MAX" "CREB1" "NR3C1" "JUN" "REST" "CEBPB" "MAX" "MYC" "CREB1" "JUN" "REST" "MAX" "REST" "CEBPB" "MAX" "FOSL1" "CEBPB")
# we have until HepG2 and CREB1
for i in "${!cell_lines[@]}"; do
  cell_line="${cell_lines[$i]}"
  tf="${tfs[$i]}"
  echo $cell_line $tf
  python3 scan/03-sample_window.py --modality epigenome --target "$tf" --cell_type "$cell_line" --window_sizes $window_sizes --step_sizes $step_sizes
done

# for these ['NR3C1', 'JUN', 'REST', 'CEBPB', 'MAX', 'FOXA1', 'MYC', 'CREB1', 'FOSL1'] tfs,
# (their profiles are [MA0113.3, MA0488.1, MA0138.2, MA0466.3, MA0058.3, MA0148.4, MA0147.3, MA0018.4, MA0477.2])
# do:
# wget http://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022/hg38/$profile.tsv.gz
# ./scan/util_scans2bigBed
# python3 scan/03-sample_window.py --modality motif --target $tf --window_sizes $window_sizes --step_sizes step_sizes --temperature 200 --offset 800

# for these cell lines: ['A549', 'HepG2', 'K562', 'GM12878', 'MCF-7', 'WTC11', 'H1', 'SK-N-SH', 'HCT116', 'IMR-90']
# do:
# python3 scan/03-sample_window.py --modality epigenome --target ATAC-seq --cell_type $cell_line --window_sizes $window_sizes --step_sizes step_sizes --temperature 200 --offset 800

# for these cell line-tf combination:
# do: 
# python3 scan/03-sample_window.py --modality epigenome --target $tf --cell_type $cell_line --step_sizes step_sizes --temperature 200 --offset 800
