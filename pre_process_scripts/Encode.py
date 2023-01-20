"""
Module for encoding categorical data in a dataframe
encode() main function
"""

import pandas as pd

def encode(df, one_hot_cols, orders={}):
    """
    Encode non-numeric columns in a dataframe
    :param df: Dataframe to encode
    :param one_hot_cols: List of columns to one-hot encode
    :param orders: Dictionary of column names with ordinal data, and the order to be encoded in
        e.g. {"col_name": ["low", "medium", "high"]}
    :return: Encoded dataframe
    """
    cols = detect_non_numeric()[0]
    for col in cols:
        if col in one_hot_cols:
            df = one_hot(col)
        elif col in orders:
            df = encode_ordinal(col, orders[col])
        else:
            df = encode_ordinal(col)
    
    return df
        

def detect_non_numeric(df):
    """
    Detect columns that are not numeric,
    :param df: Dataframe
    :return: (List of non-numeric columns, list of columns with mixed numeric and non-numeric values)
    """
    non_numeric_cols = []
    mixed_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            non_numeric_cols.append(col)
        else:
            try:
                a = df[col].astype(float)
            except ValueError:
                mixed_cols.append(col)
    return non_numeric_cols, mixed_cols


def encode_ordinal(df, col_name, order=None):
    """
    Encode ordinal values
    :param df: Dataframe
    :param col_name: Column name
    :param order: List of values in order (optional)
    :return: Encoded dataframe
    """
    if order is None:
        df[col_name] = df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
    else:
        # df[col_name] = df[col_name].astype('category', categories=order)
        df[col_name] = pd.Categorical(df[col_name], categories=order, ordered=True)
        df[col_name] = df[col_name].cat.codes
    return df


def one_hot(df, col_name):
    """
    One-hot encode a column
    :param df: Dataframe
    :param col_name: Column name
    :return: Encoded dataframe
    """
    df = pd.get_dummies(df, columns=[col_name])
    return df

