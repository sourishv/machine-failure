from matplotlib import pyplot as plt
import seaborn as sns

def display_data_info(df):
    # Viewing methods
    print("First few rows of the DataFrame:")
    print(df.head())

    print("\nDataFrame shape:")
    print(df.shape)

    print("\nDataFrame info:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nColumn names:")
    print(df.columns)

    print("\nData types of each column:")
    print(df.dtypes)

    print("\nNumber of missing values in each column:")
    print(df.isnull().sum())

    print("\nValue counts for the 'target' column:")
    print(df['prediction'].value_counts())

    # Visualization
    print("\nPairplot:")
    sns.pairplot(df)
    plt.show()

    print("\nHeatmap of the correlation matrix:")
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()