# USADAE: Autoencoder-Based Tool for RNA-seq Confounder Separation

## Description
USADAE is an autoencoder-based tool using adversarial learning to separate hidden confounding factors in RNA-seq data, which can be used for correcting confounding factors in downstream analyses.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/chenxuya/USADAE.git
   cd USADAE
   pip install -e .
   ```

## Usage
get help information
```bash
usadae-train --help
```
run USADAE
```bash
# generate simulation data
python ./simulation/sim.py 

# train
usadae-train --gene_expression_file ./simulation/data/sim_1000_100_20_all_gene_expression.txt \
             --confounder_file ./simulation/data/sim_1000_100_20_all_confounders.txt \
             --outdir ./simulation/result \
             --latent_dim 12 \
             --stage1_epochs 100 \
             --stage2_epochs 100 \
             --stage3_epochs 300 \
             --log1p \
             --lambda_adv 1 \
             --lambda_tissue 5 \
             --hidden_dims 512 256
```
## Output
- **out_prefix_bio_latent.txt**: The estimated latent biological latent representation.
- **out_prefix_conf_latent.txt**: The estimated confounder factors.
- **out_prefix_ori_recon.txt**: The reconstructed gene expression matrix.
- **out_prefix_corrected_recon.txt**: The corrected gene expression matrix.
Due to the limitation of sample size, we recommend using the confounding factors from **conf_latent.txt** as covariates in subsequent analyses (e.g., differential analysis, eQTL analysis).