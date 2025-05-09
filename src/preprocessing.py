import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.drop_duplicates()
    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    return df
