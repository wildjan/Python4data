# ### US Baby Names 1880-2010

# In[ ]:

#from __future__ import division
from numpy.random import randn
from IPython.core.magics import config
import numpy as np
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 5))
np.set_printoptions(precision=4)
get_ipython().magic(u'pwd')


# http://www.ssa.gov/oact/babynames/limits.html

# In[ ]:

#get_ipython().system(u'more names\\yob1880.txt')

# In[ ]:

import pandas as pd
names1880 = pd.read_csv('names/yob1880.txt', names=['name', 'sex', 'births'])
names1880


# In[ ]:

names1880.groupby('sex').births.sum()


# In[ ]:

# 2010 is the last available year right now
years = range(1880, 2011)

pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)


# In[ ]:

total_births = names.pivot_table('births', index='year',
                                 columns='sex', aggfunc=sum)


# In[ ]:

total_births.tail()


# In[ ]:

total_births.plot(title='Total births by sex and year')

# In[ ]:

def add_prop(group):
    # Integer division floors
    births = group.births.astype(float)

    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)


# In[ ]:

names


# In[ ]:

np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)


# In[ ]:

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)


## In[ ]:

#pieces = []
#for year, group in names.groupby(['year', 'sex']):
#    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
#top1000 = pd.concat(pieces, ignore_index=True)


# In[ ]:

top1000.index = np.arange(len(top1000))


# In[ ]:

top1000


# ### Analyzing naming trends

# In[ ]:

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']


# In[ ]:

total_births = top1000.pivot_table('births', index='year', columns='name',
                                   aggfunc=sum)
total_births


# In[ ]:

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=False, figsize=(12, 10), grid=False,
            title="Number of births per year")


# #### Measuring the increase in naming diversity

# In[ ]:

plt.figure()


# In[ ]:

table = top1000.pivot_table('prop', index='year',
                            columns='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',
           yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))


# In[ ]:

df = boys[boys.year == 2010]
df


# In[ ]:

prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
prop_cumsum[:10]


# In[ ]:

prop_cumsum.values.searchsorted(0.5)


# In[ ]:

df = boys[boys.year == 1900]
in1900 = df.sort_values(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1


# In[ ]:

def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')


# In[ ]:

def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()


# In[ ]:

diversity.plot(title="Number of popular names in top 50%")


# #### The "Last letter" Revolution

# In[ ]:

# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index=last_letters,
                          columns=['sex', 'year'], aggfunc=sum)


# In[ ]:

subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()


# In[ ]:

subtable.sum()


# In[ ]:

letter_prop = subtable / subtable.sum().astype(float)


# In[ ]:

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',
                      legend=False)


# In[ ]:

plt.subplots_adjust(hspace=0.25)


# In[ ]:

letter_prop = table / table.sum().astype(float)

deny_ts = letter_prop.ix[['d', 'e', 'n', 'y'], 'M'].T
deny_ts.head()


# In[ ]:

plt.close('all')


# In[ ]:

deny_ts.plot()


# #### Boy names that became girl names (and vice versa)

# In[ ]:

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like


# In[ ]:

filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()


# In[ ]:

table = filtered.pivot_table('births', index='year',
                             columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()


# In[ ]:

plt.close('all')


# In[ ]:

table.plot(style={'M': 'k-', 'F': 'k--'})


