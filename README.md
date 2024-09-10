# Finezyme
language modeling for enzyme sequences

## Setup
Typical installation time on the tested hardware: 5 to 10 minutes.
1. Python version tested: 3.11.4 (Tested on Ububtu OS, Kernel version: 5.15.0-94-generic, Architecture: x86_64 (64-bit))
2. Hardware requirements: GPU with more than 40GB of memory for training. GPU with more than 10GB of memory for Inference. Inference can work without GPU hardware.
3. Pip install requirements.txt file
4. Download the pre-trained foundational model from this link: `https://drive.google.com/drive/folders/1_odDCoRF35LmdTZH-bS6JICiedcqIlRs?usp=share_link`

## Files dscription and functionalities
- `generation_manager.py`: This module defines the GeneratorManager class, that can be used to generate new molechules with a model checkpoint. The generation can start with optional keywords and aminoacids prefix from where to start the generation. The module contains three generation functions:
  1) teacher forcing generation: given a sequence, it computes the resulting generation for each position in the sequence given all the previous (input) positions.
  2) after-n generation: given a sequence, the model takes the first n aminoacids as prefix and generates until the length of the actual (real) protein in input to the model. (in addition the probabilities for each amino acid predicted)
  3) generation_complete_sequence: generates sequences with optional keywords and aminoacids prefix in input, the generation is stopped when the stop keyword is generated.
- `model_manager.py`: This module contains the classes used to create the model structure (togheter with `pytorch_transformer.py` module), and handles model checkpoint loading.
- `tokenizer.py`: This module defines the tokenizer class, that allows to transform amino acids and keywords into model tokens using mapping files from `mapping_files/` directory.
- `ProteinDataset.py`: This module contains the ProteinDataset class, that is in charge (toghether with `transformProtein.py`) of loading pickle files that contain the dataset (generated from  `dataset_manager.py` module) used to train the model.
- `pytorch_training.py`: This module handles model training (Trainer class).
- `generation_and_finetuning_tutorial.ipynb`: Notebook tutoria that explains how to generate new syntetic molechules from a model checkpoint, and how to fine-tune the foundational model with any protein dataset from UniProt.
- `dataset_manager.py`: This module provides functions to load and analyze a Uniprot tsv file for training the foundational starting model.
- `blosum/`: This directory contains the blosum substitution matrix used to compute the soft accuracy measure.
- `notebooks/`: This directory contains notebooks used for subsequent analysis of generated molecules, and generation use cases.

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
- On Phage Lysozymes, the control code for phage is 0, and the stop token is 1.
- The Galaxy workflow to perform analisys on the lysozyme generated sequences is aviable at this link: `https://usegalaxy.eu/u/marco_nino/w/blast-on-generated-sequences-clustering-on-generated-sequences-original-msa-on-clustering-output-phylogenetic-tree-construction`
