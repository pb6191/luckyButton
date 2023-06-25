# %% load modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import ttest_rel, ttest_ind


pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%%

df1 = pd.read_csv("../data/raw/luckybutton_June 25, 2023_09.53_numeric.csv")

df = df1.drop(labels=[0,1], axis=0)

#%%

df.to_csv("../data/clean/clean_data.csv", index=False)
df = pd.read_csv("../data/clean/clean_data.csv")
df.dtypes

df[["condition"]].value_counts()

df["freq"].mean()
df["know"].mean()
df["freq2"].mean()
df["hover"].mean()

df[(df.condition == "deterministic")]["hover"].mean()
df[(df.condition == "probabilistic")]["hover"].mean()

ttest_rel(df['freq'], df['freq2'])
ttest_rel(df['freq2'], df['hover'])

ttest_ind(df[(df.condition == "deterministic")]['hover'], df[(df.condition == "probabilistic")]['hover'])




#%%




fig, ax = plt.subplots(figsize=(15, 8))

# Create a list of colors for the boxplots based on the number of features you have
boxplots_colors = ['yellowgreen', 'olivedrab', 'yellowgreen']

dflist = df[["freq", "freq2", "hover"]].T.values.tolist()

# Boxplot data
bp = ax.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['thistle', 'orchid', 'thistle']

# Violinplot data
vp = ax.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['tomato', 'darksalmon', 'tomato']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(features, y, s=.3, c=scatter_colors[idx])

plt.yticks(np.arange(1,4,1), ['at baseline', 'after learning\nwhat it does', 'if there was\na preview\nupon hovering'])  # Set text labels.
plt.xlabel('')
plt.xlim([0, 8])
ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(["Never", "Very\nrarely", "Rarely", "Occasionally", "Frequently", "Very\nfrequently", "Always"])
plt.title("Frequency of utilizing the 'I am feeling lucky' button")
fig.tight_layout()
fig.savefig("../figures/freq.png")
plt.show()








#%%



