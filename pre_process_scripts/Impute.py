def impute():
    return complete_case()

def complete_case(df):
    """
    Complete case analysis
    :param df: Dataframe
    :return: Dataframe with complete cases
    """
    df = df.dropna()
    return df

#Average imputation
def average_impute(df):
    """
    Impute missing values with the mean/mode
    :param df: Dataframe
    :return: Dataframe with imputed values
    """
    for col in df.columns:
        #if numeric
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean(), inplace=True)
        #if categorical
        elif df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df
