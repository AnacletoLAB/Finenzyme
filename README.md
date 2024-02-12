# ProGen
language modeling for protein sequences

## Setup
1. Python version tested: 3.11.4
2. Pip install requirements file
3. To download the pre-trained general model: `gdown https://drive.google.com/drive/folders/1_odDCoRF35LmdTZH-bS6JICiedcqIlRs?usp=share_link -O ckpt --folder`
4. To download test predictions of the general model: `gdown https://drive.google.com/drive/folders/1xgR0dj2iqgqKjZQ3KiDSNNVojgkFMsQ_?usp=share_link -O results -- folder`
5. To download test predictions of the Lysozyme fine-tuned model: `gdown https://drive.google.com/drive/folders/1g9WBp6mnU_K4guD5jFx6v0Qb2HgRmyt2?usp=share_link -O results_after_Fine_tuning --folder`


## Pretraining Vocabulary
The categories for lines in `vocab.txt`:
- 0 to 1164: keyword ids
- 1165 to 129380: taxonomy ids
- 129381 to 129405: amino acids
- 129406: PAD token

## Fine-tuned lysozyme model vocabulary
Assumptions:
- there are k clusters replacing ctrl codes [0,k-1]
- there is a stop token replacing ctrl code k
- the sample length is 511. all extra tokens are replaced with the original pad token 129406

In our case we trained specifically on Phage Lysozyme, it's ProGen code is 0, and the stop token is 1.

## Galaxy workflow
The Galaxy workflow to perform analisys on the generated sequences is aviable at this link: `https://usegalaxy.eu/u/marco_nino/w/blast-on-generated-sequences-clustering-on-generated-sequences-original-msa-on-clustering-output-phylogenetic-tree-construction`
