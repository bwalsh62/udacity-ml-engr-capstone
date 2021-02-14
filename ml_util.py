
#%% Import libraries

import pandas as pd

#%% One-Hot Encoding for DataFrames

def df_one_hot_encode(df, col):
    # Generate encoding
    oh_encoding = pd.get_dummies(df[col])

    # Concatenate new columns to dataframe
    df = pd.concat([df, oh_encoding], axis=1)

    # Drop original feature
    df = df.drop(col, axis=1)
    
    return df

