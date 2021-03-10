# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

#this user defined function PCS
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

import os
datapath = os.path.join("datasets", "lifesat", "")

# To plot pretty figures directly within Jupyter
#matplotlib inline
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Download the data. do not if I want to use 2021 oecd data
#import urllib.request
#DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
#os.makedirs(datapath, exist_ok=True)
#for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
#    print("Downloading", filename)
#    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
#    urllib.request.urlretrieve(url, datapath + filename)

# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import sklearn.neighbors
import sklearn.linear_model

# Load the data
print("loading 2015 oecd data")#dion change when using the 2021 file
print("loading old gdp per capita file")#dion change
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data by calling the user defined function PCS
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.savefig('C:/Users/dvanoverdijk/AppData/Local/Programs/Python/Python38/images/fundamentals/out.pdf', dpi=300)#turns out that order is vital must do save befor show
plt.show()

# Select a linear model
#model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 3)
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print()
print(model.predict(X_new)) # outputs [[ 5.96242338]] becomes [[6.25984414]] when using data from 2021 file

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "fundamentals"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()        
    plt.savefig(path, format=fig_extension, dpi=resolution)

#can't seem to call save_fig like plt.save_fig("out.png"), this will result in error saying that matplotlib.pyhon does not have save_fig
#it looks like this is probably supposed to be in the matplotlib.pyplot library


