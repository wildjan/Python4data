# ## MovieLens 1M data set

# In[ ]:

import pandas as pd
import os

path =  r'C:\\Users\\jwild\\Source\\Repos\\Python4data\\ch02'
os.chdir(path)

upath = os.path.expanduser('movielens/users.dat')
rpath = os.path.expanduser('movielens/ratings.dat')
mpath = os.path.expanduser('movielens/movies.dat')

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

users = pd.read_csv(upath, sep='::', header=None, names=unames)
ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames)
movies = pd.read_csv(mpath, sep='::', header=None, names=mnames)


# In[ ]:

users[:5]


# In[ ]:

ratings[:5]


# In[ ]:

movies[:5]


# In[ ]:

ratings


# In[ ]:

data = pd.merge(pd.merge(ratings, users), movies)
data


# In[ ]:

data.ix[0]


# In[ ]:

mean_ratings = data.pivot_table('rating', index='title',
                                columns='gender', aggfunc='mean')
mean_ratings[:5]


# In[ ]:

ratings_by_title = data.groupby('title').size()


# In[ ]:

ratings_by_title[:5]


# In[ ]:

active_titles = ratings_by_title.index[ratings_by_title >= 250]


# In[ ]:

active_titles[:10]


# In[ ]:

mean_ratings = mean_ratings.ix[active_titles]
mean_ratings


# In[ ]:

mean_ratings = mean_ratings.rename(index={'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)':
                           'Seven Samurai (Shichinin no samurai) (1954)'})


# In[ ]:

top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
top_female_ratings[:10]


# ### Measuring rating disagreement

# In[ ]:

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']


# In[ ]:

sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:15]


# In[ ]:

# Reverse order of rows, take first 15 rows
sorted_by_diff[::-1][:15]


# In[ ]:

# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
# Filter down to active_titles
rating_std_by_title = rating_std_by_title.ix[active_titles]
# Order Series by value in descending order
rating_std_by_title.sort_values(ascending=False)[:10]


