import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sn.set(style="white")
sn.set(style="whitegrid", color_codes=True)

data = pd.read_csv("final_cleaned_spotify.csv")
df = pd.DataFrame(data)

#correlation
correlation_matrix = df.corr()
correlation_matrix = correlation_matrix.drop(correlation_matrix.columns[[range(15)]], axis=1)
print (correlation_matrix)

min_diff = st.norm.ppf(.975)

matrix_shape = correlation_matrix.shape
height = matrix_shape[0]
width = matrix_shape[1]

significance_top_100 = 0
significance_pop_artist = 0
total_features = height

sn.heatmap(correlation_matrix, annot=True)
plt.show()

#logistic regression
top_7_features = df.drop(df.columns[[0, 1, 4, 6, 8, 10, 11, 12, 14, 16, 17, 18, 19]], axis=1)
top100 = df[['Top100']]
logreg = sm.Logit(top100, top_7_features).fit()
print(logreg.summary())

acousticness = df['acousticness']
liveness = df['liveness']
first_interaction = smf.ols(formula='top100 ~ acousticness * liveness', data=df).fit()
print(first_interaction.summary())

top_7_and_pop = df.drop(df.columns[[0, 1, 4, 6, 8, 10, 11, 12, 14, 16, 17, 18]], axis=1)
logreg = sm.Logit(top100, top_7_and_pop).fit()
print(logreg.summary())

