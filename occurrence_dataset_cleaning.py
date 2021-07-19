#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None) # lots of columns, need to set to display max


# In[2]:


occurrence_df = pd.read_csv('occurrence.txt', delimiter = "\t")
print(occurrence_df.shape)
occurrence_df.head()


# In[3]:


# getting percentages of rows with missing data in each column
def get_perc_nan_df (df):
    #pnan = occurrence_df.isnull().sum().sort_values(ascending=False)/len(occurrence_df)*100
    cnan = occurrence_df.isnull().sum().sort_values(ascending=False)
    pnan_df = cnan.to_frame()
    pnan_df = pnan_df.reset_index()
    pnan_df.columns = ['column', 'count_nan']
    pnan_df['percentage_nan'] = (pnan_df['count_nan']/len(occurrence_df)*100)
    return pnan_df


# In[4]:


# if percentage of null values in a given column is > 90%, removing it from the dataset
perc_nan_df = get_perc_nan_df(occurrence_df)
above_90_nan = perc_nan_df[perc_nan_df['percentage_nan'] > 90]
columns_to_remove = list(above_90_nan['column'])
occurrence_df = occurrence_df.drop(columns_to_remove, axis = 1)
print(occurrence_df.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(occurrence_df.columns)


# In[5]:


# removing out-of-scope columns
out_of_scope_columns = ['license', 'modified', 'rights', 'rightsHolder', 'institutionCode', 'collectionCode', 'datasetName',
                       'basisOfRecord', 'catalogNumber', 'occurrenceStatus', 'verbatimEventDate', 'coordinateUncertaintyInMeters', 
                       'dateIdentified', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'specificEpithet',
                       'taxonRank', 'taxonomicStatus', 'datasetKey', 'publishingCountry', 'lastInterpreted', 'issue', 
                       'hasCoordinate', 'hasGeospatialIssues', 'taxonKey', 'acceptedTaxonKey', 'kingdomKey', 'phylumKey',
                       'classKey', 'orderKey', 'familyKey', 'genusKey', 'speciesKey', 'genericName', 'acceptedScientificName',
                       'verbatimScientificName', 'protocol', 'lastParsed', 'lastCrawled', 'repatriated', 'level0Gid',
                       'level0Name', 'level1Gid', 'level1Name', 'level2Gid']
occurrence_df = occurrence_df.drop(out_of_scope_columns, axis = 1)

# renaming columns
occurrence_df = occurrence_df.rename({'level2Name': 'county', 'decimalLatitude': 'lat',
                                     'decimalLongitude': 'lng'}, axis=1)
print(occurrence_df.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(occurrence_df.columns)


# In[6]:


# doing check on percentage NaN in remaining columns
perc_nan_df = get_perc_nan_df(occurrence_df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(perc_nan_df)


# In[7]:


# county and state are null in 0.1 and 0.05% of rows, respectively. can use reverse geocoding to get this information
occurrence_df.loc[:, 'lat'] = occurrence_df['lat'].astype(float).round(6)
occurrence_df.loc[:, 'lng'] = occurrence_df['lng'].astype(float).round(6)
occurrence_df.loc[:, 'coordinates'] = '(' + occurrence_df['lat'].map(str) + ',' + occurrence_df['lng'].map(str) + ')'

state_na = occurrence_df[occurrence_df['stateProvince'].isna()]
county_na = occurrence_df[occurrence_df['county'].isna()]

occurrence_df.head()


# In[8]:


from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import re
geolocator = Nominatim(user_agent="geoapiExercises")

def get_geo_info (df, to_update):
    
    df.loc[:, 'address'] = df.apply(
        lambda x: geolocator.reverse((x['lat'], x['lng'])), axis=1)
    
    addresses = df['address'].to_list()
    
    state_list = []
    county_list = []
    
    for i in range(len(addresses)):
        location = addresses[i]
        if location is not None:
            address = location.raw['address']
            county = address.get('county', '')
            m = re.search('(.+?)\sCounty', county)
            if m:
                county = m.group(1)
            else:
                county = 'NaN'
            state = address.get('state', '')
            county_list.append(county)
            state_list.append(state)
        else:
            # geopy does not always return values successfully, unfortunately some will still be N/A
            county_list.append('')
            state_list.append('')
    
    if to_update == 'state':
        df['stateProvince'] = state_list
        df['stateProvince'] = df['stateProvince'].replace('',np.NaN)
    elif to_update == 'county':
        df['county'] = county_list
        df['county'] = df['county'].replace('',np.NaN)
    return df 


# In[9]:


# pass in what to update
state_df_filled = get_geo_info(state_na, 'state')
county_df_filled = get_geo_info(county_na, 'county')
county_df_filled.head()


# In[10]:


state_lookup_dict = pd.Series(state_df_filled.stateProvince.values,index=state_df_filled.gbifID).to_dict()

occurrence_df.loc[occurrence_df.gbifID.isin(state_df_filled.gbifID), 'stateProvince'] = state_df_filled['stateProvince']
occurrence_df.loc[occurrence_df.gbifID.isin(county_df_filled.gbifID), 'county'] = county_df_filled['county']


# In[11]:


# verbatimLocality no longer needed
occurrence_df = occurrence_df.drop(['verbatimLocality'], axis = 1)
# doing another check on percentage NaN in remaining columns
perc_nan_df = get_perc_nan_df(occurrence_df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(perc_nan_df)


# In[12]:


# investigating unique values in categorical columns to identify typos, non-uniform data, and remove invalid data
unique_mediaType = occurrence_df['mediaType'].value_counts() 
unique_mediaType


# In[13]:


unique_lifeStage = occurrence_df['lifeStage'].value_counts() 
unique_lifeStage


# In[14]:


unique_stateProvince = occurrence_df['stateProvince'].value_counts() 
unique_stateProvince


# In[15]:


unique_day = pd.unique(occurrence_df['day'])
unique_month = pd.unique(occurrence_df['month'])
unique_year = pd.unique(occurrence_df['year'])
print(np.sort(unique_day))
print(np.sort(unique_month))
print(np.sort(unique_year))


# In[16]:


unique_taxonID = occurrence_df['taxonID'].value_counts() 
unique_taxonID


# In[17]:


unique_scientificName = occurrence_df['scientificName'].value_counts() 
unique_scientificName


# In[18]:


unique_species = occurrence_df['species'].value_counts() 
unique_species


# In[19]:


# adding a column for sentiment analysis of comments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer= SentimentIntensityAnalyzer()
remarks_non_null = occurrence_df[occurrence_df['occurrenceRemarks'].notna()]
remarks_non_null['sentiment_analysis'] = [analyzer.polarity_scores(x)['compound'] for x in remarks_non_null['occurrenceRemarks']]

occurrence_df.loc[occurrence_df.gbifID.isin(remarks_non_null.gbifID), 'sentiment_analysis'] = remarks_non_null['sentiment_analysis']
occurrence_df.head()


# In[20]:


# creating an frame to output as a separate file for top 25 words used in occurrenceRemarks
# would be cool for a word cloud in Tableau
from collections import Counter
from nltk.corpus import stopwords

s = set(stopwords.words('english'))

# other words that pop up that don't contribute value
s.update(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'st', 'ive', 'th', 'httpswwwinaturalistorgobservations'])
pattern = '|'.join([r'\b{}\b'.format(w) for w in s])

remarks_non_null['occurrenceRemarksNoStops'] = remarks_non_null['occurrenceRemarks'].str.lower()
remarks_non_null['occurrenceRemarksNoStops'] = remarks_non_null['occurrenceRemarksNoStops'].str.replace('[^A-Za-z\s]', '')
remarks_non_null['occurrenceRemarksNoStops'] = remarks_non_null['occurrenceRemarksNoStops'].str.replace(pattern, '')
most_common_remark_words = Counter(" ".join(remarks_non_null["occurrenceRemarksNoStops"]).split()).most_common(50)

words = []
counts = []
  
for i,j in most_common_remark_words:
    words.append(i)
    counts.append(j)

top_words_df = pd.DataFrame({'word':words, 'count':counts})
top_words_df.head(50)


# In[21]:


# pulling in media to map media identifier (note only the first identifier will be mapped for simplicity)
multimedia_df = pd.read_csv('multimedia.txt', delimiter = "\t")
multimedia_df.head()


# In[22]:


occurrence_df.loc[occurrence_df.gbifID.isin(multimedia_df.gbifID), 'mediaIdentifier'] = multimedia_df['identifier']
# doing check on percentage NaN in mediaType vs identifier
perc_nan_df = get_perc_nan_df(occurrence_df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(perc_nan_df)


# In[23]:


# ensuring appropriate datatypes before exporting
occurrence_df.dtypes


# In[24]:


occurrence_df.to_csv('cleaned_occurrence_df.csv', index=False)
top_words_df.to_csv('occurrence_df_remark_word_counts_df.csv', index=False)


# In[ ]:




