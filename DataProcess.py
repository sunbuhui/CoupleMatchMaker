import pandas as pd
import _pickle as pickle
import numpy as np
from scipy.stats import halfnorm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

with open("profiles.pkl",'rb') as fp:
    df = pickle.load(fp)

p = {}

# Movie Genres
movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.28,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01,
               0.01]

# TV Genres
tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.30,
           0.23,
           0.12,
           0.12,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

# Religions (could potentially create a spectrum)
religion = ['Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Other',
            'Agnostic',
            'Atheist']

p['Religion'] = [0.16,
                 0.16,
                 0.01,
                 0.19,
                 0.11,
                 0.05,
                 0.10,
                 0.09,
                 0.07,
                 0.06]

# Music
music = ['Rock',
         'HipHop',
         'Pop',
         'Country',
         'Latin',
         'EDM',
         'Gospel',
         'Jazz',
         'Classical']

p['Music'] = [0.30,
              0.23,
              0.20,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.02]

# Sports
sports = ['Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.34,
               0.30,
               0.16,
               0.13,
               0.04,
               0.03]

# Politics (could also put on a spectrum)
politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]

# Social Media
social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

# Age (generating random numbers based on half normal distribution)
age = halfnorm.rvs(loc=18,scale=8, size=df.shape[0]).astype(int)

# Lists of Names and the list of the lists
categories = [movies, religion, music, politics, social, sports, age]

names = ['Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Age']

combined = dict(zip(names, categories))

for name, cats in combined.items():
    if name in ['Religion', 'Politics']:
        # Picking only 1 from the list
        df[name] = np.random.choice(cats, df.shape[0], p=p[name])

    elif name == 'Age':
        # Generating random ages based on a normal distribution
        df[name] = cats
    else:
        # Picking 3 from the list
        try:
            df[name] = list(np.random.choice(cats, size=(df.shape[0], 1, 3), p=p[name]))
        except:
            df[name] = list(np.random.choice(cats, size=(df.shape[0], 1, 3)))

        df[name] = df[name].apply(lambda x: list(set(x[0].tolist())))

df['Religion'] = pd.Categorical(df.Religion, ordered=True,
                                categories=['Catholic',
                                            'Christian',
                                            'Jewish',
                                            'Muslim',
                                            'Hindu',
                                            'Buddhist',
                                            'Spiritual',
                                            'Other',
                                            'Agnostic',
                                            'Atheist'])

df['Politics'] = pd.Categorical(df.Politics, ordered=True,
                                categories=['Liberal',
                                            'Progressive',
                                            'Centrist',
                                            'Moderate',
                                            'Conservative'])

# with open("refined_profiles.pkl",'wb') as fp:
#     pickle.dump(df, fp)
df.show()