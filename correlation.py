import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data = pd.read_csv("spotify_billboard_data.csv")

df = pd.DataFrame(data)

print(df)

correlation_matrix = df.corr()
correlation_matrix = correlation_matrix.drop(correlation_matrix.columns[[range(15)]], axis=1)
print (correlation_matrix)
sn.heatmap(correlation_matrix, annot=True)
plt.show()