# Machine Failure ML Predictor
from datasets import load_dataset
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# Load dataset from huggingface as dictionary
dataset = load_dataset("pgurazada1/machine-failure-mlops-demo-logs")
# Convert to Pandas DataFrame
dataset = pd.DataFrame(dataset)

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

dataset.head()
