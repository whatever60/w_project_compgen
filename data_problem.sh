# redirect all output to ./logs/data_problem.log

cell_lines="A549 A549 A549 A549 A549 HepG2 HepG2 HepG2 HepG2 HepG2 HepG2 HepG2 K562 K562 K562 K562 K562 K562 K562 K562 K562 GM12878 GM12878 GM12878 GM12878 GM12878 MCF-7 MCF-7 MCF-7 MCF-7 MCF-7 MCF-7 WTC11 WTC11 WTC11 WTC11 SK-N-SH SK-N-SH HCT116 HCT116 HCT116 HCT116 IMR-90"
tfs="NR3C1 JUN REST CEBPB MAX FOXA1 NR3C1 JUN MAX MYC CREB1 FOSL1 FOXA1 REST CEBPB MAX MYC FOSL1 CREB1 NR3C1 JUN REST CEBPB MAX MYC CREB1 FOXA1 REST CEBPB MAX CREB1 JUN MAX CREB1 NR3C1 JUN REST MAX REST CEBPB MAX FOSL1 CEBPB"
python3 data_problem.py --cell_types $cell_lines --tfs $tfs
    

