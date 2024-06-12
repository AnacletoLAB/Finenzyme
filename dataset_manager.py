import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

#seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class DatasetManager:
    def __init__(self, tsv_path, name):
        # name: identifier for the dataset
        # tsv path: path of downloaded tsv from uniprot of the protein dataset
        self.tsv_path = tsv_path
        self.name = name

    def load_data(self, verbose=False, single_key=True):
        """
        Load the Uniprot tsv data from the directory
        """
        directory = self.tsv_path
        name = self.name
        dataframe = pd.read_csv(directory, sep = "\t")
        
        #dataframe = dataframe.dropna(subset = ["EC number"]) #drop rows with missing EC number
        if verbose:
            print("check sequences: ", len(dataframe))
        dataframe = dataframe.dropna(subset = ["Sequence"]) 
        dataframe = dataframe.dropna(subset = ["Protein existence"])
        if verbose:
            print("check sequences after: ", len(dataframe))
            print("The following data regard the family with EC number: ", name)
    
        dictionary_existence = {"Evidence at protein level": 5, 
                                "Evidence at transcript level": 4, 
                                "Inferred from homology": 3, 
                                "Predicted": 2, 
                                "Uncertain": 1}
        dataframe["Protein existence"] = dataframe["Protein existence"].str.strip().map(dictionary_existence)
        dataframe["Sequence"] = dataframe["Sequence"].apply(str)
    
        #keep only such columns
        dataframe = dataframe[["Entry","EC number", "Protein existence", "Sequence", "Length", 'Gene Ontology (molecular function)']]
        if verbose:
            print("The number of sequences before removing duplicates is: ", len(dataframe))
        before = len(dataframe)
        # Sort the DataFrame by 'Sequence' for grouping and 'Protein existence' descending to have the highest on top
        dataframe = dataframe.sort_values(by=['Sequence', 'Protein existence'], ascending=[True, False])
        # Drop duplicates based on the 'Sequence' column, keeping the first entry (the one with the highest 'Existence level') for each sequence
        dataframe = dataframe.drop_duplicates(subset=['Sequence'], keep='first')
        if verbose:
            print("The number of sequences after removing duplicates is: ", len(dataframe), 'duplicates: ', before - len(dataframe))
        
        if single_key:
            dataframe["EC number"] = dataframe["EC number"].apply(lambda x: [0])
        else:
            dataframe["EC number"] = dataframe["EC number"].apply(lambda x: [0] + x) #put 0 at the beginning of the list
            dataframe["EC number"] = dataframe["EC number"].apply(lambda x: list(set(x))) #for each y take only the unique values
        
        #count EC numbers
        ec_numbers = []
        for i in dataframe["EC number"]:
            ec_numbers += i
        values, counts = np.unique(np.array(ec_numbers), return_counts = True)
        counts_dict = dict(zip(values, counts))
        dataframe["EC number"] = dataframe["EC number"].apply(lambda x: [y for y in x if counts_dict[y] > 100]) #take only EC numbers with more than 100 proteins
        sorted_list = sorted([int(x) for x in counts_dict.keys() if counts_dict[x] > 100])
        mapping_dict = dict(zip(sorted_list, range(len(sorted_list))))
        dataframe["EC number"] = dataframe["EC number"].apply(lambda x: [mapping_dict[y] for y in x]) #map the EC numbers to a range of their length
        if verbose:
            print("The number of sequences per subfamily is: ", counts_dict)
            print("The dictionary to map subfamily to keyword:", mapping_dict)
        mean_length = dataframe["Length"].mean()
        std_length = dataframe["Length"].std()
        if verbose:
            print("The mean length of the sequences is: ", round(mean_length, 3))
            print("The standard deviation of the length of the sequences is: ", round(std_length, 3))
        self.dataframe = dataframe
        self.mapping_dict = mapping_dict
        print('Dataset loaded succesfully')

    def plot_histogram(self, path, bin_step=50):
        """
        Plot and save the histogram of the lengths of the sequences
        """
        if not os.path.exists(path):
            os.makedirs(path)
        data = self.dataframe["Length"].to_numpy()
        # Create the histogram
        max_val = max(data)
        bin_edges = list(range(0, 500, bin_step))
        #print(bin_edges)
        plt.figure(figsize=(11, 6))
        n, bins, _ = plt.hist(data, bins=bin_edges, color=(180/255, 211/255, 178/255), edgecolor='black', alpha=0.99, rwidth=0.87)    
        #print("n:", n)
        #print("bins:", bins)
        # Title, labels, and legend
        plt.title('Histogram of Sequence Length Ranges for EC ' + self.name)
        plt.xlabel('Sequence Length')
        plt.ylabel('Number of Instances')
        #plt.legend()
        _ = plt.show()
        plt.savefig(path+"histogram_"+self.name+'.svg')

    def split_train_test_validation_old(self, train_fraction, validation_fraction, seed=SEED):
        dataframe = self.dataframe
        mapping_dict = self.mapping_dict
        # Split the dataframe into training, validation, and test sets
        dataframe["subfamily"] = dataframe["EC number"].apply(lambda x: x[-1])
        train_dataframes = []
        validation_dataframes = []
        test_dataframes = []
        for i in mapping_dict.values():
            sub_dataframe = dataframe[dataframe["subfamily"] == i]
            train_dataframe = sub_dataframe.sample(frac=train_fraction, random_state=seed)
            remaining_dataframe = sub_dataframe.drop(train_dataframe.index)
            validation_fraction_adjusted = validation_fraction / (1 - train_fraction)  # Adjusting fraction based on remaining data
            validation_dataframe = remaining_dataframe.sample(frac=validation_fraction_adjusted, random_state=seed)
            test_dataframe = remaining_dataframe.drop(validation_dataframe.index)
            train_dataframes.append(train_dataframe)
            validation_dataframes.append(validation_dataframe)
            test_dataframes.append(test_dataframe)
        
        train = pd.concat(train_dataframes).drop(columns=["subfamily"])
        validation = pd.concat(validation_dataframes).drop(columns=["subfamily"])
        test = pd.concat(test_dataframes).drop(columns=["subfamily"])
        self.train_db = train
        self.test_db = test
        self.validation = validation

    def split_train_test_validation(self, train_fraction, validation_fraction, test_fraction, seed=SEED):
        dataframe = self.dataframe
        mapping_dict = self.mapping_dict
        
        # Ensure the fractions sum up to 1
        assert train_fraction + validation_fraction + test_fraction == 1, "dataset fractions must sum up to 1"
    
        # Split the dataframe into training, validation, and test sets
        dataframe["subfamily"] = dataframe["EC number"].apply(lambda x: x[-1])
        train_dataframes = []
        validation_dataframes = []
        test_dataframes = []
        
        for i in mapping_dict.values():
            sub_dataframe = dataframe[dataframe["subfamily"] == i]
            
            # Split into training data
            train_dataframe = sub_dataframe.sample(frac=train_fraction, random_state=seed)
            remaining_dataframe = sub_dataframe.drop(train_dataframe.index)
            
            # Split remaining data into validation and test data
            validation_dataframe = remaining_dataframe.sample(frac=validation_fraction/(validation_fraction + test_fraction), random_state=seed)
            test_dataframe = remaining_dataframe.drop(validation_dataframe.index)
            
            train_dataframes.append(train_dataframe)
            validation_dataframes.append(validation_dataframe)
            test_dataframes.append(test_dataframe)
        
        train = pd.concat(train_dataframes).drop(columns=["subfamily"])
        validation = pd.concat(validation_dataframes).drop(columns=["subfamily"])
        test = pd.concat(test_dataframes).drop(columns=["subfamily"])
        
        self.train_db = train
        self.validation = validation
        self.test_db = test
    
    def convert_to_dict(self, dataframe):
        #convert the dataframes to dictionaries
        dictionary = {}
        for _, row in dataframe.iterrows():
            sub_dict = {}
            sub_dict["kw"] = row["EC number"]
            sub_dict["ex"] = row["Protein existence"]
            sub_dict["seq"] = row["Sequence"]
            sub_dict["len"] = row["Length"]
            sub_dict['GO_molecular_function'] = row['Gene Ontology (molecular function)']
            dictionary[row["Entry"]] = sub_dict
        # convert the dictionary items to a list, shuffle the list, and convert it back to a dictionary
        items = list(dictionary.items())
        random.shuffle(items)
        return dict(items)
    
    def save_to_pickle(self, path):
        name=self.name
        # if dir does not exiast create it
        if not os.path.exists(path):
            os.makedirs(path)
        
        #save the dictionary to a pickle file
        train = self.convert_to_dict(self.train_db)
        test = train = self.convert_to_dict(self.test_db)
        validation = train = self.convert_to_dict(self.validation)
        
        with open(path+f'training_{name}.p', "wb") as file:
            pickle.dump(train, file)
        with open(path+f'test_{name}.p', "wb") as file:
            pickle.dump(test, file)
        with open(path+f'validation_{name}.p', "wb") as file:
            pickle.dump(validation, file)
        self.pickle_path = path

    def save_to_fasta(self, path_to_save):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        name = self.name
        path_to_load = self.pickle_path
        # training
        with open(path_to_load+f'training_{name}.p', "rb") as file:
            data = pickle.load(file)
        with open(path_to_save +f'training_{name}.fasta', "w") as f:
            for seq_id, seq_info in data.items():
                f.write(f">{seq_id}\n{seq_info['seq']}\n")
        # validation
        with open(path_to_load+f'validation_{name}.p', "rb") as file:
            data = pickle.load(file)
        with open(path_to_save +f'validation_{name}.fasta', "w") as f:
            for seq_id, seq_info in data.items():
                f.write(f">{seq_id}\n{seq_info['seq']}\n")
        # test
        with open(path_to_load+f'test_{name}.p', "rb") as file:
            data = pickle.load(file)
        with open(path_to_save +f'test_{name}.fasta', "w") as f:
            for seq_id, seq_info in data.items():
                f.write(f">{seq_id}\n{seq_info['seq']}\n")

if __name__ == "__main__":
    # data_specific_enzymes/databases/tsvs/name.tsv
    directory = "data_specific_enzymes/databases/tsvs/uniprotkb_ec_3_2_1_4_AND_length_10_TO_5_2024_04_29.tsv"
    name = "ec_3_2_1_4"
    datasetManager = DatasetManager(directory, name)
    datasetManager.load_data()
    
    path_hist = "data_specific_enzymes/databases/tmp/"
    datasetManager.plot_histogram(path_hist)
    
    #split the dataframes into training, validation, and test sets for each EC number 90% training, 5% test, 5% validation
    datasetManager.split_train_test_validation(train_fraction=0.9, validation_fraction=0.05)
    
    #convert the dataframes to dictionaries, and save them as pickles
    datasetManager.save_to_pickle("data_specific_enzymes/databases/pickles_test/")

    with open("data_specific_enzymes/databases/pickles_test/training_"+name+ ".p", "rb") as file:
        training = pickle.load(file)
    keys = list(training.keys())
    for key in keys[:10]:
        print(key, training[key])

    datasetManager.save_to_fasta("data_specific_enzymes/databases/fasta_test/")
    