# Machine Failure ML Predictor
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

# Load dataset from huggingface as dictionary
dataset = load_dataset("pgurazada1/machine-failure-mlops-demo-logs")
# Convert to Pandas DataFrame
df = pd.DataFrame(dataset)
# Split df into multiple columns
# 'Type' (str) is L (light, ex: drill machine), M (medium, ex: mill, cnc),
# or H (heavy, ex: multi-axis cnc) machine type or l,m,h size lathes, etc
#
print(df.head())
df[['Air Temperature (K)', 'Process Temperature (K)', 'Rotational Speed (RPM)',
    'Torque (Nm)', 'Tool Wear (min)', 'Type', 'prediction']] = pd.DataFrame(df['train'].tolist(), index=df.index)

# Split the DataFrame 80% training, 20% test (42 is seed for reproducibility)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Removes rows with missing values (4 missing tool types found)
df.dropna(how='all', inplace=True)

# formatting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Training Set:")
print(train_df)
print("\nTest Set:")
print(test_df)

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
