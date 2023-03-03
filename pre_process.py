"""
Control the pre-processing of the data.
"""
import yaml
import sys
from operator import itemgetter
import pandas as pd
from sklearn.model_selection import train_test_split 

import pre_process_scripts.encode as encode
import pre_process_scripts.impute as impute
import pre_process_scripts.outlier_removal as outlier_removal
import pre_process_scripts.feature_selection as feature_selection
import phases as phases

#Load config file path from command line
config_path = sys.argv[1]

def load_config(path):
    with open(path) as cf_file:
        config = yaml.safe_load( cf_file.read() )

    return config

def pipeline(train, test, config, ret_phases=False):
    """
    Control the pre-processing of the data.
    :param train: training data
    :param test: test data
    :param config: config file
    :return: pre-processed train and test data
    """
    #Remove outliers
    train, test = outlier_removal.router(train, test, config['outlier_removal'])

    #Imputation
    train, test = impute.router(train, test, config['imputation'])

    scores =[]
    if ret_phases:
        #for each row in test
        for i in range(len(test)):
            row = test.iloc[i]
            scores.append(phases.construct(row))

    #Encode
    one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    train = encode.encode(train, one_hot_cols, orders)
    test = encode.encode(test, one_hot_cols, orders)
    train, test = encode.onehot_align(train, test)

    #Feature selection
    train, test = feature_selection.router(train, test, config['feature_selection'])

    return train, test, scores


def main():
    """
    Run with a config file defining data split and method etc
    Writes split, pre-processed data to files
    """
    config = load_config(config_path)

    #Load data
    file_path = config['data']['input']
    df = pd.read_csv(file_path)

    #Drop column "ID4Sasan"
    if "ID4Sasan" in df.columns:
        df.drop("ID4Sasan", axis=1, inplace=True)
    
    #Split data
    test_size, random_state = itemgetter('test_size', 'random_state')(config["split"])
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    #Pre-process data
    train, test = pipeline(train, test, config)

    #Output data
    if config["data"]["output"] is not None:
        train.to_csv(config["data"]["output"]["train"], index=False)
        test.to_csv(config["data"]["output"]["test"], index=False)


if __name__ == '__main__':
    main()