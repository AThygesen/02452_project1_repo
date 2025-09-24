import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

import seaborn as sns
import matplotlib.pyplot as plt

# Plotting style (from exercises)
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.)

# This is taken form the internet (get a reference). Make sure the working directory is the script directory.
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


#### Taken from the exercise 2 ####
# Number of attributes
M = X.shape[1]

# Plot a matrix scatter plot of the wine attributes, colored by the wine color
fig, axs = plt.subplots(M, M, figsize=(20, 20), sharex='col', sharey='row')
for i in range(M):
    for j in range(M):
        for color in y.unique(): # loop through each label
            # Construct a mask based on the label
            mask = (y == color)
            # Plot the scatter plot for attribute pair (if not on the diagonal)
            axs[i, j].scatter(
                x=X[mask].iloc[:, j],        # x-values for the $j$'th attribute
                y=X[mask].iloc[:, i],        # y-values for the $i$'th attribute
                label=color, alpha=0.3,
                color='r' if color == 'Red' else 'y'
            )

# Update titles
for col in range(M):
    axs[0, col].set_title(X.columns[col])
    axs[col, 0].set_ylabel(X.columns[col])

# Add the legend to the last subplot only
axs[0,0].legend(loc='upper left')
plt.tight_layout(pad=1.)
plt.show()

####################################