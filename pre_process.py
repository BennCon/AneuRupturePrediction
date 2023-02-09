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

    #Drop column "ID4Sasan"
    if "ID4Sasan" in df.columns:
        df.drop("ID4Sasan", axis=1, inplace=True)
    
    #Split data - TEMPORARILY doing this after imputation and encoding
    #TODO: find solution where one hot encoding doesn't leave columns in test but not in train (or vice versa)
    test_size, random_state = itemgetter('test_size', 'random_state')(config["split"])
    test, train = train_test_split(df, test_size=test_size, random_state=random_state)

    # #Remove outliers
    # train = impute.remove_outliers(train)
    # test = impute.remove_outliers(test)


    #Imputation 
    train_avgs = impute.get_data_avg(train)
    train = impute.impute_data(train, train_avgs)
    test = impute.impute_data(test, train_avgs)
    


    #Encode
    one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    train = encode.encode(train, one_hot_cols, orders)
    test = encode.encode(test, one_hot_cols, orders)

    #Intuition: any columns not in both were added by one hot encoding, so add them to the other, and fill with 0
    #Columns in test but not in train:
    test_dif = (set(test.columns) - set(train.columns))
    for col in test_dif:
        train[col] = 0
    #Columns in train but not in test:
    train_dif = (set(train.columns) - set(test.columns))
    for col in train_dif:
        test[col] = 0

    

    # df=impute.complete_case(df)
    # one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    # df=encode.encode(df, one_hot_cols, orders)


    #Feature selection
    retain_cols = feature_selection.select(train, config['feature_selection'])
    train = train[retain_cols]
    test = test[retain_cols]

    #Output data
    if config["data"]["output"] is not None:
        train.to_csv(config["data"]["output"]["train"], index=False)
        test.to_csv(config["data"]["output"]["test"], index=False)


if __name__ == '__main__':
    main()