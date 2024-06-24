# Finezyme
language modeling for enzyme sequences

## Setup
1. Python version tested: 3.11.4
2. Pip install requirements file
3. To download the pre-trained general model: `gdown https://drive.google.com/drive/folders/1_odDCoRF35LmdTZH-bS6JICiedcqIlRs?usp=share_link -O ckpt --folder`
4. To download the fine-tuned model on Phage lysozymes: `https://drive.google.com/file/d/17-_ewwXF3bTvchDh9PnB4KSxijvLOulj/view?usp=share_link`

## Finenzyme Data aviability
The Datasets used to train Finenzyme have been downloaded with UniProt. As explained in detail inside the notebook tutorial `generation_and_finetuning_tutorial.ipynb`, the Only two filters used when downloading the sequences from UniProt are discarding sequences with less than 10 amino acids and more than 500 amino acids.
Regarding the reduced test sets: in order to further test Finenzyme we computed filtered test sets based on BLASTp output, with the full test set and the training set. The filtered test contains sequences with BLASTp sequence similarity of the full test set less than 70% with respect to the training set.

## Finenzyme model vocabulary
Assumptions:
- there are k clusters replacing ctrl codes [0,k-1].
- there is a stop token replacing ctrl code k.
- the sample length is 511. all extra tokens are replaced with the original pad token 129406.

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
- On Phage Lysozymes, it's the control code for phage is 0, and the stop token is 1.
- The Galaxy workflow to perform analisys on the generated sequences is aviable at this link: `https://usegalaxy.eu/u/marco_nino/w/blast-on-generated-sequences-clustering-on-generated-sequences-original-msa-on-clustering-output-phylogenetic-tree-construction`
