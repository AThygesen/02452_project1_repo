import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

import seaborn as sns
import matplotlib.pyplot as plt
import re

# Plotting style (from exercises)
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.)

# This is taken form the internet (get a reference). Make sure the working directory is the script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# load the dataset
df = pd.read_excel('data/Concrete_data/Concrete_Data.xls')


# Select the target variable. Doing it by column number
X = df.iloc[:, :-1]
y = pd.Categorical(df.iloc[:, -1])

# Check the shape of the data (from exercises)
N, M = X.shape
assert N == 1030, "There should be 1030 samples in the Concrete dataset."
assert M == 8, "There should be 8 components in the Concrete dataset."


############### USED AI FOR THIS ########################
############## WRITE OUR OWN CODE #######################

# Boxplots of all 8 components
fig, axs = plt.subplots(1, M, figsize=(2.2*M + 2, 5))  # no shared y so each feature auto-scales
fig.suptitle("Boxplots of Concrete Dataset components", y=0.98, fontsize=12)

for idx, col in enumerate(X.columns):
	ax = axs[idx]
	col_data = X[col].values
	# Using matplotlib's boxplot for fine control
	bp = ax.boxplot(col_data, vert=True, patch_artist=True,
					boxprops=dict(facecolor=f"C{idx}", alpha=0.45, color=f"C{idx}"),
					medianprops=dict(color='black', linewidth=1.2),
					whiskerprops=dict(color=f"C{idx}"),
					capprops=dict(color=f"C{idx}"),
					flierprops=dict(marker='o', markerfacecolor=f"C{idx}", markersize=3, markeredgecolor='none', alpha=0.55))

	# Dynamic y-limits with small padding
	dmin, dmax = col_data.min(), col_data.max()
	rng = dmax - dmin
	pad = 0.05 * rng if rng > 0 else 1
	ax.set_ylim(dmin - pad, dmax + pad)

	# Build a title without any parenthetical units or component annotations
	clean_title = re.sub(r'\([^)]*\)', '', col)  # remove all (...) segments
	clean_title = re.sub(r'\s+', ' ', clean_title).strip()
	# Extract unit from column name: take the last parenthetical group that is not a component descriptor
	matches = re.findall(r'\(([^()]*)\)', col)
	unit = None
	if matches:
		for m in reversed(matches):
			if 'component' not in m.lower():
				unit = m.strip()
				break
		if unit is None:  # fallback to last
			unit = matches[-1].strip()
	ax.set_ylabel(unit if unit else "Value", fontsize=7)

plt.tight_layout()
plt.show()
##################### END OF AI CODE #####################

############## END OF OUR CODE ###########################