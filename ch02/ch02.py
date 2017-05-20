from __future__ import division
# coding: utf-8

# # Introductory examples

# ## 1.usa.gov data from bit.ly


# In[ ]:

get_ipython().magic(u'pwd')


# In[ ]:

path = 'usagov_bitly_data2012-03-16-1331923249.txt'


# In[ ]:

open(path).readline()


# In[ ]:

import json
path = 'usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]


# In[ ]:

records[0]


# In[ ]:

records[0]['tz']


# In[ ]:

print(records[0]['tz'])


# ### Counting time zones in pure Python

# In[ ]:

time_zones = [rec['tz'] for rec in records]


# In[ ]:

time_zones = [rec['tz'] for rec in records if 'tz' in rec]


# In[ ]:

time_zones[:10]


# In[ ]:

def get_counts(sequence):
    counts = {}
    for elm in sequence:
        counts[elm] = counts.get(elm, 0) + 1
    return counts


# In[ ]:

from collections import defaultdict

def get_counts2(sequence):
    counts = defaultdict(int) # values will initialize to 0
    for x in sequence:
        counts[x] += 1
    return counts


# In[ ]:

counts = get_counts(time_zones)


# In[ ]:

counts['America/New_York']


# In[ ]:

len(time_zones)


# In[ ]:

def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


# In[ ]:

top_counts(counts)


# In[ ]:

from collections import Counter


# In[ ]:

counts = Counter(time_zones)


# In[ ]:

counts.most_common(10)


# ### Counting time zones with pandas


# In[ ]:

#from __future__ import division
#from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4)


# In[ ]:

import json
path = 'usagov_bitly_data2012-03-16-1331923249.txt'
lines = open(path).readlines()
records = [json.loads(line) for line in lines]


# In[ ]:

from pandas import DataFrame, Series
import pandas as pd

frame = DataFrame(records)
frame


# In[ ]:

frame['tz'][:10]


# In[ ]:

tz_counts = frame['tz'].value_counts()
tz_counts[:10]


# In[ ]:

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10]


# In[ ]:

plt.figure(figsize=(10, 4))


# In[ ]:

tz_counts[:10].plot(kind='barh', rot=0)


# In[ ]:

frame['a'][1]


# In[ ]:

frame['a'][50]


# In[ ]:

frame['a'][51]


# In[ ]:

results = Series([x.split()[0] for x in frame.a.dropna()])
results[:5]


# In[ ]:

results.value_counts()[:8]


# In[ ]:

cframe = frame[frame.a.notnull()]


# In[ ]:

operating_system = np.where(cframe['a'].str.contains('Windows'),
                            'Windows', 'Not Windows')
operating_system[:5]


# In[ ]:

by_tz_os = cframe.groupby(['tz', operating_system])


# In[ ]:

agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]


# In[ ]:

# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer[:10]


# In[ ]:

count_subset = agg_counts.take(indexer)[-10:]
count_subset


# In[ ]:

plt.figure()


# In[ ]:

count_subset.plot(kind='barh', stacked=True)


# In[ ]:

plt.figure()


# In[ ]:

normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
