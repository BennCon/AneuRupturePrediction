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

def construct_process(train, config):
    """
    Given training data, create the transformations to be applied to both train and test
    :param train: training data
    :param config: config file
    """
    #Outlier Removal
    outlier_removal_data = outlier_removal.generate(train, config['outlier_removal'])
    train_copy = outlier_removal.apply(train, outlier_removal_data)


    #Encoding
    one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    encoding_data = {'one_hot_cols': one_hot_cols, 'orders': orders}
    train_copy = encode.encode(train_copy, one_hot_cols, orders)
    #Imputation
    imputation_data = impute.generate(train_copy, config['imputation'])
    train_copy = impute.apply(train_copy, imputation_data)

    #Feature Selection
    retain_cols = feature_selection.router(train_copy, config['feature_selection'])
    # print(f"Retained {len(retain_cols)} features")

    return {
        'outlier_removal': outlier_removal_data,
        'imputation': imputation_data,
        'encoding': encoding_data,
        'feature_selection': retain_cols
    }

def apply_process(df, data):
    df = outlier_removal.apply(df, data['outlier_removal'])
    df = encode.encode(df, data["encoding"]["one_hot_cols"], data["encoding"]["orders"])
    df = impute.apply(df, data['imputation'])

    #Encode

    #Feature Selection
    for i in data["feature_selection"]:
        if i not in df.columns:
            df[i] = 0
    df = df[data["feature_selection"]]

    return df


def pipeline(train, config, test=None, ret_phases=False):
    """
    Control the pre-processing of the data.
    :param train: training data
    :param test: test data
    :param config: config file
    :return: pre-processed train and test data
    """
    #Remove outliers
    if test is not None:
        train, test = outlier_removal.generate(train, config['outlier_removal'], test)
    else:
        train = outlier_removal.generate(train, config['outlier_removal'])[0]

    #Imputation
    if test is not None:
        train, test = impute.generate(train, config['imputation'], test)
    else:
        train = impute.generate(train, config['imputation'])[0]

    scores = []
    if ret_phases:
        #for each row in test
        for i in range(len(test)):
            row = test.iloc[i]
            scores.append(phases.construct(row))

    #Encode
    one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    train = encode.encode(train, one_hot_cols, orders)
    if test is not None:
        test = encode.encode(test, one_hot_cols, orders)
        train, test = encode.onehot_align(train, test)

    #Feature selection
    if test is not None:
        train, test = feature_selection.router(train, config['feature_selection'], test)
    else:
        train = feature_selection.router(train, config['feature_selection'])[0]

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