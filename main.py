# Machine Failure ML Predictor
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_visualization import display_data_info


# Load dataset from huggingface as dictionary
dataset = load_dataset("pgurazada1/machine-failure-mlops-demo-logs")
# Convert to Pandas DataFrame
df = pd.DataFrame(dataset)
# Split df into multiple columns
# 'Type' (str) is L (light, ex: drill machine), M (medium, ex: mill, cnc),
# or H (heavy, ex: multi-axis cnc) machine type or l,m,h size lathes, etc

#print(df.head())
df[['Air Temperature (K)', 'Process Temperature (K)', 'Rotational Speed (RPM)',
    'Torque (Nm)', 'Tool Wear (min)', 'Type', 'prediction']] = pd.DataFrame(df['train'].tolist(), index=df.index)

# Convert 'Type' to one-hot encoding:

# Define mapping: Quality variants L/M/H = 2/3/5 additional minutes tool wear
size_mapping = {'L': 2, 'M': 2, 'H': 5}

# Apply mapping to your data
df['Type'] = df['Type'].map(size_mapping)

# Split the DataFrame 80% training, 20% test (42 is seed for reproducibility)
#DO THIS AFTER PREPROCESSING
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Removes rows with missing values (4 missing tool types found)
df.dropna(how='all', inplace=True)

# Remove outliers
df = df[(df['Process Temperature (K)'] < 400) & (df['Process Temperature (K)'] > 111)]
# Remove outliers for torque
df = df[(df['Torque (Nm)'] > 5) & (df['Torque (Nm)'] < 150)]

# Formatting:

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#print("Training Set:")
#print(train_df)
#print("\nTest Set:")
#print(test_df)

# Visualization: Seaborn plot commented out in data_visualization.py (opens in new window)
# display_data_info(df)

# PREPROCESSING LAYERS: For now, only normalization and buckets
# Normalization:

# Define the normalization layer
# Extract the relevant columns for normalization
features = ['Air Temperature (K)', 'Process Temperature (K)', 'Rotational Speed (RPM)', 'Torque (Nm)', 'Tool Wear (min)']

# Initialize a dictionary to store normalization layers for each feature
normalization_layers = {}

# Create and adapt normalization layers for each feature
for feature in features:
    # Define the normalization layer
    norm_layer = tf.keras.layers.Normalization(axis=None)
    # Adapt the normalization layer to the feature data
    norm_layer.adapt(df[feature].values.reshape(-1, 1))
    # Store the normalization layer in the dictionary
    normalization_layers[feature] = norm_layer

# Print the type of normalization layers to verify
for feature, layer in normalization_layers.items():
    print(f"{feature}: {type(layer)}")

# Concatenate our inputs into a single tensor. (reshape: to 2d array with 1 column)
preprocessing_layers = tf.keras.layers.Concatenate()(
    [normalization_layers[feature](df[feature].values.reshape(-1, 1)) for feature in features])