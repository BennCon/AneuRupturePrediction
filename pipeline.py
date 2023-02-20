"""
This controls the pipeline of the project.
And allows use of cross validation.
"""
from sklearn.metrics import accuracy_score
import yaml
import sys
import pandas as pd

import pre_process as pre_process
import pre_process_scripts.impute as impute
import classify as classify

import classifiers.knn as knn
import classifiers.logisticRegression as lr

#Load config file path from command line
config_path = sys.argv[1]

def load_config(path):
    with open(path) as cf_file:
        config = yaml.safe_load( cf_file.read() )

    return config

def pred(model, x_test, classifier):
    if classifier == 'kNN':
        y_pred = knn.classify(model, x_test)
    elif classifier == 'logisticRegression':
        y_pred = lr.classify(model, x_test)

    return y_pred

def main():
    config = load_config(config_path)

    #Load data
    df = pd.read_csv(config['data'])

    #Drop column "ID4Sasan"
    if "ID4Sasan" in df.columns:
        df.drop("ID4Sasan", axis=1, inplace=True)

    pre_process_config_path = config['pre_process_config']
    pre_process_config = load_config(pre_process_config_path)

    #If imputation is set to complete_case, then we need to do that before splitting into folds 
    if pre_process_config['imputation']['method'] == 'complete_case':
        df = impute.complete_case(df)

    #K fold cross validation
    if config["loo"]:
        k = len(df)
    else:
        k = config['kfold']
    fold_size = int(len(df)/k)
    folds = [df[i*fold_size:(i+1)*fold_size] for i in range(k)]
    
    #Loop through the folds - each time using pre_process.pipeline passing train and test
    accs = []
    for i in range(k):
        train = pd.concat([folds[j] for j in range(k) if j != i])
        test = folds[i]

        #Convert train and test so they are not slices of the original data frame
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)

        #Progress bar for each fold that automatically updates in the terminal
        print(f"Fold {i+1}/{k}", end="\r")
        #Format it as a bar chart with a percentage
        print(f"Fold {i+1}/{k} [{'='*int((i+1)/k*20)}{' '*(20-int((i+1)/k*20))}] {int((i+1)/k*100)}%", end="\r")

        train, test = pre_process.pipeline(train, test, pre_process_config)


        #Train model
        classifier_config_path = config['classifier_config']
        classifier_config = load_config(classifier_config_path)
        model = classify.pipeline(train, classifier_config)

        #Classify test data
        x_test, y_test = test.drop('ruptureStatus', axis=1), test['ruptureStatus']
        y_pred = pred(model, x_test, classifier_config['classifier'])

        #Accuracy
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

        

    print("\n")
    print("Accuracy: {}".format(sum(accs)/len(accs)))

if __name__ == '__main__':
    main()