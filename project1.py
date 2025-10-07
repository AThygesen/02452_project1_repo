import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Plotting style
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.)

# This is taken form the internet (get a reference). Make sure the working directory is the script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# load the dataset
df = pd.read_excel('data/Concrete_data/Concrete_Data.xls')

# Clean column names ## GET REFERENCE FOR THIS ##
df.columns = [col.split(' (')[0] for col in df.columns]

# Select the target variable. Doing it by column number
X = df.iloc[:, :-1]
y = pd.Categorical(df.iloc[:, -1])





######### SECTION FOR HANDLING OUTLIERS #########
# COMPONENT NAMES 

# 'Cement (component 1)(kg in a m^3 mixture)'
# 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)'
# 'Fly Ash (component 3)(kg in a m^3 mixture)'
# 'Water  (component 4)(kg in a m^3 mixture)'
# 'Superplasticizer (component 5)(kg in a m^3 mixture)'
# 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'
# 'Fine Aggregate (component 7)(kg in a m^3 mixture)'
# 'Age (day)'
# 'Concrete compressive strength(MPa, megapascals)'

mask = (X['Blast Furnace Slag'] < 0.00001) | (X['Fly Ash'] < 0.00001) | (X['Superplasticizer'] < 0.00001)

# Remove outliers with conditional filtering
#X = X[~mask]
#y = y[~mask]


#######################################################

# Check the shape of the data (from exercises)
# N, M = X.shape
# assert N == 1030, "There should be 1030 samples in the Concrete dataset."
# assert M == 8, "There should be 8 components in the Concrete dataset."


############### BOXPLOT & SUMMARY STATISTICS ###############

# Plot a boxplot of the attributes in X
#X.plot(kind='box', subplots=True, layout=(3, 3), figsize=(22,10), sharex=False, sharey=False)
#plt.show()

# Compute summary statistics
#print(X.describe())

###########################################################


# ######### CORRELATION MATRIX FROM LECTURE 3 #########

# # Transform the target variable into a numerical format
# y_numerical = y.codes
# # Convert y_numerical to a pandas Series with a name
# y_numerical_series = pd.Series(y_numerical, index=X.index, name="Target")
# # Construct the modified dataframe
# df_tilde = pd.concat([X, y_numerical_series], axis=1)
# # Compute the correlation matrix
# correlation_matrix = df_tilde.corr()

# # Plot the correlation matrix
# fig = plt.figure(figsize=(12, 24))
# fig.suptitle('Correlation matrix of the standardized data', fontsize=16)
# plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
# plt.colorbar()
# plt.xticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Concrete compressive strength"], rotation=90)
# plt.yticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Concrete compressive strength"])
# plt.grid(False)

# #### FIND REFERENCE FOR THIS ####
# # Print correlation values in each box
# for i in range(correlation_matrix.shape[0]):
# 	for j in range(correlation_matrix.shape[1]):
# 		plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)



# plt.show()

# ################################

#mask = (df['Blast Furnace Slag'] < 0.00001) | (df['Fly Ash'] < 0.00001) | (df['Superplasticizer'] < 0.00001)
#df = df[~mask]

# Creating figures manually gives some flexibility that is nice when 
# you have a specific layout in mind. This is nice for reports or publications.
# fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharey=False)
# fig.suptitle("Histograms of dataset components", fontsize=16)
# for j in range(9):
# 	data = df.iloc[:, j]
# 	axs[j // 3, j % 3].hist(data, color=f"C{j}", bins=20, edgecolor='black')
# 	axs[j // 3, j % 3].set_title(df.columns[j])
# 	axs[j // 3, j % 3].set_xlabel("Value")
# 	axs[j // 3, j % 3].set_ylabel("Number of cases")
# 	# Adjust y-axis to fit the data tightly
# 	counts, _ = np.histogram(data, bins=20)
# 	axs[j // 3, j % 3].set_ylim(0, counts.max() * 1.1)
# plt.tight_layout()
# plt.show()

# ################################

# Number of attributes
M = X.shape[1]

 # Convert y to float (if not already)
strength = y.astype(float)
strength_class = pd.qcut(strength, q=3, labels=['Low', 'Medium', 'High'])

# Use the strength_class defined earlier in the script
labels = ['Low', 'Medium', 'High']
bins = pd.qcut(strength, q=3, retbins=True)[1]
legend_labels = [
	f"{label} [{bins[i]:.2f}, {bins[i+1]:.2f})"
	for i, label in enumerate(labels)
]

# Plot a matrix scatter plot of the attributes, colored by strength class
fig, axs = plt.subplots(M, M, figsize=(20, 20), sharex='col', sharey='row')
colors = {'Low': 'b', 'Medium': 'g', 'High': 'r'}
for i in range(M):
	for j in range(M):
		for idx, cls in enumerate(labels):
			mask = (strength_class == cls)
			X_filtered = X.loc[mask]
			axs[i, j].scatter(
				x=X_filtered.iloc[:, j],
				y=X_filtered.iloc[:, i],
				label=legend_labels[idx] if (i == 0 and j == 0) else None,
				alpha=0.3,
				color=colors[cls],
				s=8
			)
		# Set axis labels
		if i == M - 1:
			axs[i, j].set_xlabel(X.columns[j], fontsize=8)
		if j == 0:
			axs[i, j].set_ylabel(X.columns[i], fontsize=8)

# Add the legend to the first subplot
axs[0, 0].legend(
	title="Strength Class",
	loc='upper left',
	fontsize=6,
	title_fontsize=6
)
plt.tight_layout(pad=2.0)
plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
plt.show()
