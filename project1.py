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
X = X[~mask]
y = y[~mask]

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


######### CORRELATION MATRIX FROM LECTURE 3 #########

# Transform the target variable into a numerical format
y_numerical = y.codes
# Convert y_numerical to a pandas Series with a name
y_numerical_series = pd.Series(y_numerical, index=X.index, name="Target")
# Construct the modified dataframe
df_tilde = pd.concat([X, y_numerical_series], axis=1)
# Compute the correlation matrix
correlation_matrix = df_tilde.corr()

# Plot the correlation matrix
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Correlation matrix of the standardized data', fontsize=16)
plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Concrete compressive strength"], rotation=90)
plt.yticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Concrete compressive strength"])
plt.grid(False)

#### FIND REFERENCE FOR THIS ####
# Print correlation values in each box
for i in range(correlation_matrix.shape[0]):
	for j in range(correlation_matrix.shape[1]):
		plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)

################################

plt.show()


########## END OF CODE ##########