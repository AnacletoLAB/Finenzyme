'''
This module implemets ProGen generation as a classifier, comprising two main classes:
1) teacher forcing generation: given a sequence, it computes the resulting generation for eache position in the sequence given all the previous (real) positions. (in addition the probabilities for each amino acid predicted)
2) after-n generation: given a sequence, the model takes the first n aminoacids as prefix and generates until the length of the actual (real) protein in input to the model. (in addition the probabilities for each amino acid predicted)
'''

