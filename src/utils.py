def basic_overview(df):
    print("Shape of dataset:", df.shape)
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

def class_distribution(df):
    print("\nDiagnosis distribution:\n", df['diagnosis'].value_counts())
