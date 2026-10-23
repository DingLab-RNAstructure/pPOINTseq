# pPOINT-seq

1. We start from indexed non-poly(A) BAMs. `src/classify_reads/classify_bam_splicing_readthrough.py` writes one read table per sample: `results/read_through/<sample>.elongating.classify.tsv`.
2. Scripts for the analyses below are in `src/classify_reads`, `src/splicing_ratio`, `src/overall_splicing`, `src/first_intron`, `src/readthrough`, and `src/cold_cui`. `src/project_config.py` is shared.
3. Data, BAMs, and `results/` are not included. This folder is the project root for `src/project_config.py`, so reference files belong in `data/refs` and BAMs in `data/bams`.
4. Python dependencies are listed in `requirements.txt`: `pandas`, `numpy`, `scipy`, `matplotlib`, `plotnine`, `seaborn`, `statsmodels`, `pysam`, and `IPython`.

## Folder structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── samples.tsv
└── src/
    ├── _provenance.py
    ├── bootstrap.py
    ├── project_config.py
    ├── classify_reads/
    │   └── classify_bam_splicing_readthrough.py
    ├── cold_cui/
    │   ├── _find_nv_consec_c5_down.py
    │   ├── _plot_nv_consec_c5_down.py
    │   ├── analyze_consecutive_intron_retention.py
    │   ├── count_consecutive_intron_retention.py
    │   ├── plot_consecutive_ri_barcode.py
    │   ├── plot_consecutive_ri_p3_offset.py
    │   ├── plot_consecutive_ri_run_length_all_reads.py
    │   ├── plot_consecutive_ri_summary.py
    │   ├── test_consecutive_ri.py
    │   └── test_consecutive_ri_read_sample.py
    ├── first_intron/
    │   ├── build_first_intron_expression.py
    │   ├── build_intron1_length_table.py
    │   ├── correlation_splicing_expression.py
    │   └── first_splicing_dynamics.py
    ├── overall_splicing/
    │   ├── first_last_min3_coordination.py
    │   ├── multiple_intron_splicing_dynamics.py
    │   ├── plot_gene_reads.py
    │   ├── polii_feature_splicing_ratio.py
    │   ├── splicing_pairs.py
    │   └── test_first_last_coupling.py
    ├── readthrough/
    │   └── plot_classified_gene_reads.py
    └── splicing_ratio/
        └── classify_splicing_tables.py
```

## Step 1: Classify each read with the splicing states

- `src/classify_reads/classify_bam_splicing_readthrough.py` reads one sorted, indexed BAM and the exon/intron annotation.
- Only primary, same-strand alignments with MAPQ at least 20 are kept.
- An intron is spliced when a CIGAR `N` covers at least 80% of the annotated intron. The junction is canonical when both splice sites are within 5 bp of the annotation; otherwise it is also reported as alternative splicing. Both count as spliced.
- An intron is retained when one continuous aligned block covers the intron plus 10 bp on each side, or when the read 5' end lies inside that intron more than 10 bp past its 5' splice site and CIGAR has not already called it spliced.
- `covered_introns` is the intron span between the read 5' and 3' ends. `spliced_introns` and `retained_introns` come from the CIGAR and 5'-end evidence above.
- A read is `fully_spliced` when every covered intron is spliced and none are retained. It is `fully_unspliced` when every covered intron is retained and none are spliced. A mismatch is `partial_spliced`.
- Reads whose 5' end is outside the promoter and exon 1 are kept and labelled `invalid_p5_start`. A 3' end inside an intron is recorded and does not change the splicing class.
- The same table records readthrough past the PAS (`is_readthrough`, `readthrough_dist`) and alternative-splicing labels (`SE`, `RI`, `MXE`, `A5SS`, `A3SS`).

```
python src/classify_reads/classify_bam_splicing_readthrough.py \
  --bam data/bams/Warm2.nonpolya.bam \
  --annotation data/refs/exon_intron_pos.chr.bed \
  --pas data/refs/Ath.PAS.completed.bed \
  --tolerance 5 \
  --min-junction-fraction 0.8 \
  --min-anchor 10 \
  --output results/read_through/Warm2.elongating.classify.tsv
```

final outputs:

- `results/read_through/Warm2.elongating.classify.tsv`

## Step 2: Splicing ratio calculation for individual introns

- `src/splicing_ratio/classify_splicing_tables.py` turns one classify TSV into an intron-level splicing-ratio table.
- An intron listed in `spliced_introns` is spliced. An intron listed in `retained_introns` is unspliced.
- For each gene, intron, and 3'-end feature, `spliced_ratio` is spliced reads divided by spliced plus unspliced reads.
- Reads whose 5' end is outside the promoter and exon 1 stay out of the table. Pass `--include-invalid-p5` to count them.
- The scripts in the later steps call the same counting themselves.

```
python src/splicing_ratio/classify_splicing_tables.py \
  --classify results/read_through/Warm2.elongating.classify.tsv \
  --output results/splicing/Warm2.elongating.splicing_ratio.tsv
```

final outputs:

- `results/splicing/Warm2.elongating.splicing_ratio.tsv`

## Step 3: Splicing ratio dynamics with the distance past each intron's 3' splice site

- `src/overall_splicing/multiple_intron_splicing_dynamics.py` plots the spliced percentage of intron positions 1–9 and 10+ against how far the read 3' end has passed that intron's 3' splice site.
- Reads start in the promoter or exon 1.
- Bins are 100 bp out to 2000 bp, and a bin needs at least 20 intron calls.
- Warm.mean averages bins present in at least two replicates.

```
python src/overall_splicing/multiple_intron_splicing_dynamics.py
```

final outputs:

- `results/splicing_order/*.multiple_intron_splicing_dynamics.pdf`

## Step 4: Example genes for visualisation

- `src/overall_splicing/plot_gene_reads.py` draws reads from one or more indexed BAM files on the gene model in `data/refs/exon_intron_pos.chr.bed`.
- A read with a spliced-alignment gap is red. A read with no gap is blue when its start and end fall in different features, and gray when both ends fall in the same exon or intron.
- Pass one `--label` per BAM, in the same order as `--bam`.
- `--bed` can be omitted; the default is `data/refs/exon_intron_pos.chr.bed`. Without `--out-dir`, the files are written next to the first BAM, in a `gene_reads_figures` directory.

```
python src/overall_splicing/plot_gene_reads.py \
  --gene-id AT3G09500.1 \
  --bam data/bams/Warm2.nonpolya.bam \
  --label Warm2 \
  --bed data/refs/exon_intron_pos.chr.bed \
  --out-dir results/gene_reads
```

final outputs:

- `results/gene_reads/AT3G09500.1_Warm2_gene_reads.pdf`

## Step 5: Splicing ratio by the exon or intron that holds the 3' end

- `src/overall_splicing/polii_feature_splicing_ratio.py` counts each target intron from `spliced_introns` and `unspliced_introns`, including reads whose 5' end is outside the promoter and exon 1.
- Bins with more than 10 reads are kept. Warm.mean averages the replicate ratios.
- The boxplot facets the feature that holds the Pol II 3' end (exon or intron, index 2–6) against the target intron.
- When the 3' end is inside intron k, that intron's box is left out, and the remaining boxes are the other introns at that position. The number above a box is the number of gene bins.

```
python src/overall_splicing/polii_feature_splicing_ratio.py
```

final outputs:

- `results/splicing/*.all_p5.total_exon_intron.splicing_ratio.pdf`

## Step 6: Splicing combination of adjacent splicing pairs

- `src/overall_splicing/splicing_pairs.py` counts adjacent intron pairs on the same read when one intron is spliced and the other is retained.
- It reports whether the upstream or the downstream intron is spliced, overall and split by intron number (2–9 and 10+).
- Reads start in the promoter or exon 1. Warm.mean averages the three replicate percentages.

```
python src/overall_splicing/splicing_pairs.py
```

final outputs:

- `results/splicing_order/*.splicing_pairs*.pdf`

## Step 7: Calculatation of First- and last-intron coordination 

- `src/overall_splicing/first_last_min3_coordination.py` counts reads whose first and last introns are both decided.
- Genes with at least three introns are kept so the pair is not adjacent.
- It calls `src/overall_splicing/test_first_last_coupling.py` for the odds ratio and the replicate test.

```
python src/overall_splicing/first_last_min3_coordination.py
```

final outputs:

- `results/exon_intron/min3/`

## Step 8: Dynamics changes of First-intron splicing ratio with different genomic features

- `src/first_intron/first_splicing_dynamics.py` keeps reads that cover intron 1 and whose 3' end has passed that intron's 3' splice site.
- Windows are 100 bp with a 10 bp step.
- The curve for every first intron is the mean of the three Warm replicates, with the replicate SD as the band. It runs out to 2500 bp.
- The same windows are then split by intron 1 length: shorter than 100 bp, and longer than 300 bp. That plot runs out to 1000 bp. Warm gets its own replicate-mean version.

```
python src/first_intron/first_splicing_dynamics.py
```

final outputs:

- `results/exon_intron/SS1.splicing.pdf`
- `results/exon_intron/SS1.splicing.group.pdf`
- `results/exon_intron/SS1.splicing.group.Warm.mean.pdf`

## Step 9: Correlation between First-intron length and mean spliced ratio

- A separate gene-level table joins intron 1 length with the mean spliced ratio.
- `src/first_intron/build_first_intron_expression.py` counts intron 1 in each replicate.
- `src/first_intron/build_intron1_length_table.py` averages the ratio within a group and sums the reads from the replicates that enter that average.
- `spliced_ratio` is the mean of replicate ratios. `unspliced_reads`, `spliced_reads`, and `total_reads` are sums of those same replicates, so dividing the sums does not recover `spliced_ratio`.
- Pass `--plot` to draw the length table.

```
python src/first_intron/build_first_intron_expression.py
python src/first_intron/build_intron1_length_table.py
python src/first_intron/build_intron1_length_table.py --plot
```

## Step 10: Identification of readthrough reads

- Each classify TSV records `is_readthrough` and `readthrough_dist` for every read.
- `src/readthrough/plot_classified_gene_reads.py` draws one gene and colors a read as readthrough when its 3' end passes that gene's PAS by more than `--min-readthrough-distance` (default 50 bp).
- Readthrough reads are red. Reads that stop before that distance are blue. Other reads in the locus are gray.
- The inset pie is the readthrough fraction, and two stacked bars give the intron-1 and intron-2 splice ratios in the readthrough and non-readthrough groups.

```
python src/readthrough/plot_classified_gene_reads.py \
  --gene-id AT2G41100.1 \
  --bam data/bams/Warm2.nonpolya.bam \
  --classify results/read_through/Warm2.elongating.classify.tsv \
  --color-by readthrough \
  --label Warm2 \
  --out-dir results/gene_reads
```

final outputs:

- `results/gene_reads/AT2G41100.1_Warm2_readthrough_gene_reads.pdf`
- `results/gene_reads/AT2G41100.1_Warm2_readthrough_gene_reads.summary.tsv`

## Step 11: Cold and consecutive unspliced introns analysis

- Cold is the cold-treated group. A consecutive unspliced-intron read (CUI read) has neighboring `RI:intron=k` and `RI:intron=k+1` in `alternative_splicing_details`.
- Reads with `splicing_class` `invalid_p5_start` are excluded.
- `src/cold_cui/count_consecutive_intron_retention.py` counts these reads and writes run length, the fraction of valid-P5 reads, the 3'-end offset of CUI reads, and an RI-only gene-read track.
- Tables and bars go to `results/alternative_splicing/`. The gene track goes to `results/gene_reads/`.

```
python src/cold_cui/count_consecutive_intron_retention.py
```

final outputs:

- `results/alternative_splicing/`
- `results/gene_reads/`

## Step 12: Comparisons between Warm versus Cold CUIs

- `src/cold_cui/analyze_consecutive_intron_retention.py` runs after the count.
- It compares Warm and Cold gene by gene, tests the library rate, fits a per-gene GLM, and builds the same-molecule 2×2 table.
- It draws the CUI figures: the per-gene intron barcode, the summary figure, and read tracks for genes that are high in Warm and lower in Cold.

```
python src/cold_cui/analyze_consecutive_intron_retention.py
```

## Step 13: Visualisation of CUIs in warm and cold with matched read-depth test

- `src/cold_cui/test_consecutive_ri_read_sample.py` draws the same number of valid-P5 reads from each replicate.
- It tests whether the CUI rate is higher in Warm than in Cold.

```
python src/cold_cui/test_consecutive_ri_read_sample.py
```
