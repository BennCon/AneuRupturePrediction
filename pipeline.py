"""
This controls the pipeline of the project.
And allows use of cross validation.
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split 
import yaml
import sys
import pandas as pd
import numpy as np

import pre_process as pre_process
import pre_process_scripts.impute as impute
import classify as classify

import classifiers.knn as knn
import classifiers.logisticRegression as lr
import classifiers.mlp as mlp

#Load config file path from command line
config_path = sys.argv[1]

def load_config(path):
    with open(path) as cf_file:
        config = yaml.safe_load( cf_file.read() )

    return config

def pred(model, x_test, classifier):
    y_pred = model.predict(x_test)

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

    #If imputation is set to complete_case, need to do that before splitting into folds 
    if pre_process_config['imputation']['method'] == 'complete_case':
        df = impute.complete_case(df)

    #Gets evalution method and metrics from config file
    eval = config['eval']
    metrics = config['metric']

    accs = []
    aucs = []

    if eval == 'loo': #(leave one out)
        k = len(df)
    elif eval == 'kfold':
        k = config['kfold']
        df = df.sample(frac=1).reset_index(drop=True)

    fold_size = int(len(df)/k)
    folds = [df[i*fold_size:(i+1)*fold_size] for i in range(k)]

    if eval != 'loo':
        #If not every fold has both classes, shuffle again and split into folds
        while not all([len(folds[i][folds[i]['ruptureStatus'] == 'Ruptured']) > 0 and len(folds[i][folds[i]['ruptureStatus'] == 'Unruptured']) > 0 for i in range(k)]):
            df = df.sample(frac=1).reset_index(drop=True)
            folds = [df[i*fold_size:(i+1)*fold_size] for i in range(k)]    


    accs = [] #Accuracies for each fold
    aucs = [] #Array of areas under the curve for each fold

    #AUC for leave one out - i.e. record tpr and fpr for each record
    preds = []
    true = []
    for i in range(k):
        train = pd.concat([folds[j] for j in range(k) if j != i]).copy()
        test = folds[i].copy()

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

        if eval == 'loo':
            preds.append(y_pred)
            true.append(y_test)
        else:
            accs.append(accuracy_score(y_test, y_pred))
            aucs.append(roc_auc_score(y_test, y_pred))


    
    print("\n")
    if eval == 'loo':
        #Use preds and true to calculate accuracy and AUC
        print(f"Accuracy: {accuracy_score(true, preds)}")
        print(f"AUC: {roc_auc_score(true, preds)}")
    else:
        print(f"Accuracy: {np.mean(accs)}")
        print(f"AUC: {np.mean(aucs)}")
        print(f"S.d. of accuracy: {np.std(accs)}")
        print(f"S.d. of AUC: {np.std(aucs)}")
        


if __name__ == '__main__':
    main()