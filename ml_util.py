
#%% Import libraries

import pandas as pd

#%% One-Hot Encoding for DataFrames

def df_one_hot_encode(df, col):
    # Generate encoding
    oh_encoding = pd.get_dummies(df[col])
    
    # Rename columns to be more descriptive in cases where there are many IDs
    oh_encoding.columns = [col+'_'+str(val) for val in oh_encoding.columns]

    # Concatenate new columns to dataframe
    df = pd.concat([df, oh_encoding], axis=1)

    # Drop original feature
    df = df.drop(col, axis=1)
    
    return df

