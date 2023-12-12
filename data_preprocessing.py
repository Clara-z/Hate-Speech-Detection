# store data (which is in .csv format) into a .npy file
# Input: data in .csv format
# Proces data, only keep some columns, and store the data into a .npy file
# Output: data in .npy format

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def store_data(input_file, cols_kept, output_file):
    """
    Store data (which is in .csv format) into a .npy file.
    
    Args:
        input_file: input file name
        cols_kept: columns kept
        output_file: output file name
    """
    # read the data
    data = pd.read_csv(input_file)
    # keep only some columns
    data = data[cols_kept]
    # convert the data into a numpy array
    data = data.values
    # add index to the data
    data = np.c_[np.arange(len(data)), data]
    # store the data into a .npy file
    np.save(output_file, data)

# split the data into training set, dev set, and test set
# and store them into separate .npy files
def split_data(input_file, output_train_file, output_dev_file, output_test_file):
    """
    Split the data into training set, dev set, and test set and store them into separate .npy files.
    
    Args:
        input_file: input file name
        output_train_file: output train file name
        output_dev_file: output dev file name
        output_test_file: output test file name
    """
    # read the data from the .npy file
    data = np.load(input_file, allow_pickle=True)
    # split the data into training set, dev set, and test set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.125, random_state=42)
    # store the data into separate .npy files
    np.save(output_train_file, train_data)
    np.save(output_dev_file, dev_data)
    np.save(output_test_file, test_data)
    

if __name__ == "__main__":
    cols_kept = ['text', 'label']
    store_data('data/dynamically-generated-hate-speech.csv', cols_kept,'data/hate_speech.npy')
    split_data('data/hate_speech.npy', 'data/hate_speech_train.npy', 'data/hate_speech_dev.npy', 'data/hate_speech_test.npy')
