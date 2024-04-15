#!/usr/bin/env python
# coding: utf-8

# # DATA PREPARATION

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("google_playstore_data.csv")


# In[5]:


df


# In[6]:


df.info


# In[5]:


df.head(5)


# In[4]:


df.tail(5)


# In[10]:


df.describe()


# In[12]:


# There os a null value
df.isnull().sum()


# In[13]:


df.nunique()


# In[15]:


df.duplicated().sum()


# # CATEGORY EXPLORATION

# In[12]:


category_counts = df['Category'].value_counts()
print(category_counts)


# In[11]:


import matplotlib.pyplot as plt
category_counts.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('App Distribution by Category')
plt.show()


# # METRIC ANALYSIS

# In[22]:


# Examine app ratings, size, popularity, and pricing trends
print('Minimum Rating:', df['Rating'].min())
print('Maximum Rating:', df['Rating'].max())
print('Average App Size:', df['Size'].mean(), 'MB')
print('Most Reviewed App:', df['Reviews'].max(), 'Reviews')


# # SENTIMENT ANALYSIS

# In[ ]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


# In[52]:


from textblob import TextBlob

text1 = "Rating"
blob1 = TextBlob(text1)


# In[53]:


blob1.sentiment


# In[50]:


df.shape


# In[54]:


polarity_score = []

for i in range(0, df.shape[0] ):
    score = TextBlob(df.iloc[i][1])
    score1 = score.sentiment[0]
    polarity_score.append(score1)


# In[58]:


df = pd.concat([df, pd.Series(polarity_score)], axis = 1)

# to the polarity_score column to the original DF


# In[59]:


df.head(5)


# In[60]:


df.rename(columns={df.columns[14]  :"Sentiment"}, inplace = True)


# In[37]:


df.head(5)


# In[45]:


df['sentiment'] = df['Rating'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

len(df[df.sentiment > 0])
len(df[df.sentiment < 0])
len(df[df.sentiment > .1])


# # INTERACTION VISUALIZATION

# In[1]:


import pandas as pd
df = pd.read_csv("google_playstore_data.csv")

df

import plotly.express as px
# Create an interactive scatter plot of app ratings and reviews

fig = px.scatter(df, x='Reviews', y='Rating', color='Category')
fig.update_layout(title='App Ratings vs. Reviews',
                  xaxis_title='Number of Reviews',
                  yaxis_title='Rating')
fig.show()


# # PRICING STRATEGIES

# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

# Analyze the distribution of app prices

sns.histplot(df['Price'], bins=20)
plt.xlabel('Price (USD)')
plt.ylabel('Count')
plt.title('App Price Distribution')
plt.show()


# Investigate the relationship between price and rating

plt.scatter(df['Price'], df['Rating'])
plt.xlabel('Price (USD)')
plt.ylabel('Rating')
plt.title('Price vs. Rating')
plt.show()


# # APP SIZE & PERFORMANCE 

# In[16]:


# Explore the distribution of app sizes
sns.histplot(df['Size'], bins=20)
plt.xlabel('App Size (MB)')
plt.ylabel('Count')
plt.title('App Size Distribution')
plt.show()


# Investigate the relationship between app size and rating
plt.scatter(df['Size'], df['Rating'])
plt.xlabel('App Size (MB)')
plt.ylabel('Rating')
plt.title('App Size vs. Rating')
plt.show()


# # TRENDING APPS & CATEGORIES

# In[3]:


top_rated = df.nlargest(10, 'Rating')
print(top_rated[['App', 'Rating', 'Reviews']])

top_categories = df.groupby('Category')['Reviews'].sum().nlargest(5)
print(top_categories)


# In[ ]:




