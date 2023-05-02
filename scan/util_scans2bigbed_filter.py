import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("bed_path", type=str)
parser.add_argument("chrom_size_path", type=str)

args = parser.parse_args()

# this bed file will have 6 columns
df_bed = pd.read_table(args.bed_path, header=None)
chroms = set(pd.read_table(args.chrom_size_path, header=None)[0].tolist())
# filter
df_bed = df_bed[[i in chroms for i in df_bed[0]]]
df_bed.to_csv(args.bed_path, sep="\t", header=None, index=False)


