import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

import seaborn as sns
import matplotlib.pyplot as plt

# Plotting style
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.)

# This is taken form the internet. Make sure the working directory is the script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Dataset
df = pd.read_excel('data/Concrete_data/Concrete_Data.xls')


# Select the target variable by column order
X = df.iloc[:, :-1]
y = pd.Categorical(df.iloc[:, -1])

# Check the shape of the data
N, M = X.shape
assert N == 1030, "There should be 1030 samples in the Concrete dataset."
assert M == 8, "There should be 8 features in the Concrete dataset."

# Display the first few rows of the dataframe
print(X.head())
