#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import pandas as pd


# In[137]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[138]:


movies.head()


# In[139]:


credits.head()


# In[140]:


credits.head(1)['cast'].values


# In[141]:


credits.head(1)['crew'].values


# In[142]:


movies = movies.merge(credits,on = 'title')


# In[143]:


movies.head(1)


# In[144]:


movies.info()


# In[ ]:





# In[ ]:





# In[145]:


# Genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[146]:


movies.head()


# In[147]:


movies.isnull().sum()


# In[148]:


movies.dropna(inplace=True)


# In[149]:


movies.duplicated().sum()


# In[150]:



movies.iloc[0].genres


# In[151]:


import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[152]:


movies.dropna(inplace=True)


# In[153]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[154]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[ ]:





# In[ ]:





# In[155]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[156]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[157]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[158]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[159]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[160]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[161]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[162]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[163]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[164]:


movies.head()


# In[165]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[166]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[167]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[168]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[169]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[170]:


vector = cv.fit_transform(new['tags']).toarray()


# In[171]:


vector.shape


# In[172]:


from sklearn.metrics.pairwise import cosine_similarity


# In[173]:


similarity = cosine_similarity(vector)


# In[174]:


similarity


# In[175]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[176]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[177]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[178]:


movies.head()


# In[185]:


movies['overview'] = movies['overview'].apply(lambda sublist: [word for string in sublist for word in string.split()])


# In[186]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[187]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[188]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[189]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[205]:


vector = cv.fit_transform(new['tags']).toarray()


# In[206]:


vector.shape


# In[208]:


vector[0]


# In[ ]:





# In[ ]:





# In[192]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:





# In[193]:


similarity = cosine_similarity(vector)


# In[209]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# In[ ]:





# In[194]:


similarity


# In[195]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[200]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        


# In[201]:


recommend('Gandhi')


# In[210]:


recommend('Avatar')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




