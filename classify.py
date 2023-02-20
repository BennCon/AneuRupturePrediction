"""
Control file for the classification process.
"""

import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

import classifiers.knn as knn
import classifiers.logisticRegression as lr

#Load config file path from command line
config_path = sys.argv[1]

def load_config(path):
    with open(path) as cf_file:
        config = yaml.safe_load( cf_file.read() )

    return config

def pipeline(train, config):
    """
    Trains the selected model.
    :param train: training data
    :param config: config file
    :return: trained model
    """
    x_train, y_train = train.drop('ruptureStatus', axis=1), train['ruptureStatus']

    #Train model
    classifier = config['classifier']
    params = config['classifiers'][classifier]

    if classifier == 'kNN':
        model = knn.train(x_train, y_train, params)
    elif classifier == 'logisticRegression':
        model = lr.train(x_train, y_train, params)
    
    return model


def main():
    config = load_config(config_path)

    #Load data
    train = pd.read_csv(config['data']['input']['train'])
    test = pd.read_csv(config['data']['input']['test'])

    #Split into x and y
    x_train, x_test = train.drop('ruptureStatus', axis=1), test.drop('ruptureStatus', axis=1)
    y_train, y_test = train['ruptureStatus'], test['ruptureStatus']

    #Train model
    classifier = config['classifier']
    params = config['classifiers'][classifier]

    if classifier == 'kNN':
        model = knn.train(x_train, y_train, params)
        #Classify test data
        y_pred = knn.classify(model, x_test)
    elif classifier == 'logisticRegression':
        model = lr.train(x_train, y_train, params)
        #Classify test data
        y_pred = lr.classify(model, x_test)

    
    #Write predictions to file
    pd.DataFrame(y_pred).to_csv(config['data']['output'], index=False)

    #Accuracy
    acc = accuracy(y_test, y_pred)

    print("Accuracy: {}".format(acc))



if __name__ == '__main__':
    main()