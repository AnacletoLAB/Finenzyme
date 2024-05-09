import pandas as pd
import numpy as np
import random

#MAPPING UTILS
def create_mapping_dict(kw: list):
    """
    Create a mapping dictionary for the keywords
    """
    mapping_dict = {}
    for i, k in enumerate(kw):
        mapping_dict[k] = i + 1
    return mapping_dict

def write_mapping_dict(mapping_dict: dict, file_name: str):
    """
    Write the mapping dictionary to a piclke file
    """
    import pickle
    with open(file_name, 'wb') as handle:
        pickle.dump(mapping_dict, handle)

def get_last_index(kw: list):
    """
    Get the last index of the keywords
    """
    return len(kw)

def create_aa_dict(last_index: int):
    """
    Create a mapping dictionary for the amino acids
    """
    aa_dict = {}
    alphabet = [chr(i) for i in range(65,91) if i != 74]
    for i in range(len(alphabet)):
        aa_dict[alphabet[i]] = i + last_index
    return aa_dict

def write_aa_dict(aa_dict: dict, file_name: str):
    """
    Write the mapping dictionary to a piclke file
    """
    import pickle
    with open(file_name, 'wb') as handle:
        pickle.dump(aa_dict, handle)

def write_vocab(dicts: list, path = 'mapping_files/vocab.txt'):
    """
    Write the mapping dictionary to a piclke file
    """
    #combine the dictionaries into one
    vocab = {}
    for d in dicts:
        vocab.update(d)

    #write it to a txt file in the format:
    #key value\n
    with open(path, 'w') as f:
        f.write("%s %s\n" % ('STOP', 0)) #STOP TOKEN
        for key in vocab.keys():
            f.write("%s %s\n" % (key, vocab[key]))
        f.write("%s %s\n" % ('PAD', len(vocab)))


#DATA FORMATTING UTILS
def model_dataframe(dataframe: pd.DataFrame) -> dict:
    """
    Convert a dataframe to the correct dictionary format for the model
    """

    #convert the dataframes to dictionaries
    if list(dataframe.columns).sort() != ["Entry", "Sequence", "seq_length", "kw"].sort():
        raise ValueError("The dataframe must have the following columns: Entry, Sequence, seq_length, kw")

    dictionary = {}
    for _, row in dataframe.iterrows():
        sub_dict = {}
        sub_dict["kw"] = row["kw"]
        sub_dict["ex"] = 4 #PLACEHOLDER I DONT KNOW WHAT THIS IS YET
        sub_dict["seq"] = row["seq"]
        sub_dict["len"] = row["seq_length"]
        dictionary[row["Entry"]] = sub_dict
    # convert the dictionary items to a list, shuffle the list, and convert it back to a dictionary
    items = list(dictionary.items())
    random.shuffle(items)
    return dict(items)