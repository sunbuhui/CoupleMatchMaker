import _pickle as pickle
import pandas as pd
with open("refined_profiles.pkl","rb") as fp:
    df = pickle.load(fp)

print(df)
df.to_csv('res.csv')