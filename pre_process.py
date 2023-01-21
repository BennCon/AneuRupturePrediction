"""
Control the pre-processing of the data.
"""
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split 

import pre_process_scripts.encode as encode
import pre_process_scripts.impute as impute
import pre_process_scripts.feature_selection as feature_selection

#Load config file path from command line
config_path = sys.argv[1]

def load_config(path):
    with open(path) as cf_file:
        config = yaml.safe_load( cf_file.read() )

    return config

def main():
    config = load_config(config_path)

    #Load data
    file_path = config['data']['input']
    df = pd.read_csv(file_path)

    #Split data
    test, train = train_test_split(df, test_size=0.2, random_state=42)

    #Imputation (for now complete case analysis)
    train = impute.complete_case(train)
    test = impute.complete_case(test)

    #Encode
    train = encode.encode(train, config['encoding']['one_hot_cols'], config['encoding']['orders'])
    test = encode.encode(test, config['encoding']['one_hot_cols'], config['encoding']['orders'])

    #Feature selection
    retain_cols = feature_selection.select(train, config['feature_selection'], return_cols=True)
    train = train[retain_cols]
    test = test[retain_cols]

    if config["data"]["output"] is not None:
        train.to_csv(config["data"]["output"]["train"], index=False)
        test.to_csv(config["data"]["output"]["test"], index=False)

    

if __name__ == '__main__':
    main()





