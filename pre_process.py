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
    
    #Split data
    test_size, random_state = itemgetter('test_size', 'random_state')(config["split"])
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    #Remove outliers
    outlier_z = config["outlier_z"]
    if outlier_z is not None:
        train = impute.remove_outliers(train, outlier_z)
        test = impute.remove_outliers(test, outlier_z)

    #Imputation
    train, test = impute.router(train, test, config['imputation'])

    #Encode
    one_hot_cols, orders = itemgetter('one_hot_cols', 'orders')(config['encoding'])
    train = encode.encode(train, one_hot_cols, orders)
    test = encode.encode(test, one_hot_cols, orders)
    train, test = encode.onehot_align(train, test)

    #Feature selection
    train, test = feature_selection.router(train, test, config['feature_selection'])

    #Output data
    if config["data"]["output"] is not None:
        train.to_csv(config["data"]["output"]["train"], index=False)
        test.to_csv(config["data"]["output"]["test"], index=False)


if __name__ == '__main__':
    main()