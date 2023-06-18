# progen
language modeling for protein sequences

## Setup
1. Pip install requirements file
2. Patch the `/usr/local/lib/python2.7/dist-packages/tensorflow_estimator/python/estimator/keras.py` (or equivalent, if installed elsewhere) by running 

```patch -b <path_to_tensorflow_estimator_package>/python/estimator/keras.py estimator.patch```

## Training Command
Currently you can train by running `python training.py --tfrecords_dir <PATH> --model_dir <PATH>`. 

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
