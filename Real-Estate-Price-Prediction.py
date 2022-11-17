#!/usr/bin/env python
# coding: utf-8

# # Real Estate Price Prediction in Bengaluru
# 
# Original Dataset: Dataset is downloaded from here: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data
# 
# Note: Almost all of the code belong to Codebasics and Dhaval Patel.This is not my own creation.This project was made just for studying purpose so that I get a hands-on experience and to get an intution on how data science and machine learning is used in real world.
#  
# Youtube Link: Mr.Dhaval Patel takes you through a fantastic journey of building a real estate prediction model in bengaluru using previous labeled data found in Kaggle. I have attached the link to the first video of the playlist. The subsequent videos can be found in the description. People interseted in Web development can also know how a model can be deployed in a website.
# 
# Video 1: https://youtube.com/watch?v=rdfbcdP75KI&list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg
# 

# ## Importing libraries and Loading the dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df1 = pd.read_csv('bengaluru_house_prices.csv')
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.groupby('area_type')['area_type'].agg('count')


# ## Data Cleaning

# **Droping the features that are not required to build our model**

# In[5]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis = 1)


# In[6]:


df2.shape


# In[7]:


df2.head()


# **Handling NA Values**

# In[8]:


df2.isnull().sum()


# In[9]:


df3 = df2.dropna()
df3.isnull().sum()


# In[10]:


df3['size'].unique()


# ## Feature Engineering
# 
# **Since th size column contains strings, we are trying to convert it into integer which just spefies the number of bedrooms in each house. Add new feature for bhk for ML model convinience**

# In[11]:


df3['bhk'] = df3['size'].apply(lambda x:int(x.split(' ')[0]))


# In[12]:


df3.head()


# In[13]:


df3['bhk'].unique()


# In[14]:


df3[df3.bhk>20]


# **Exploring total_sqft feature**

# In[15]:


df3.total_sqft.unique()


# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion.It is better to just drop such corner cases because it wont makeany significant difference.

# In[18]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[19]:


convert_sqft_to_num('2100-2850')


# In[20]:


convert_sqft_to_num('34.46Sq. Meter	')


# In[21]:


df4 = df3.copy()


# In[22]:


df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)


# In[23]:


df4.head(3)


# In[24]:


df4.iloc[30]


# In[25]:


df4.head()


# In[26]:


df5 = df4.copy()


# **Add new feature called price per square feet**

# In[27]:


df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[28]:


df5.location.unique()


# In[29]:


len(df5.location.unique())


# **Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations**

# In[30]:


df5.location = df5.location.apply(lambda x: x.strip())


# In[31]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[32]:


len(location_stats[location_stats<=10])


# ## Dimensionality Reduction
# 
# **Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns**
# 

# In[33]:


location_stats_less_than_10 = location_stats[location_stats<=10]


# In[34]:


location_stats_less_than_10


# In[35]:


len(df5.location.unique())


# In[36]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)


# In[37]:


len(df5.location.unique())


# In[84]:


df5.location.head(20)


# In[85]:


df5.head(20)


# ## Outlier detection and removal
# 
# **As a data scientist when you have a conversation with your business manager (who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft**

# In[40]:


df5[df5.total_sqft/df5.bhk <300].head()


# Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely

# In[41]:


df5.shape


# In[42]:


df6 = df5[~(df5.total_sqft/df5.bhk <300)]


# In[43]:


df6.shape


# In[44]:


df6.price_per_sqft.describe()


# **Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation**

# In[45]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out


# In[46]:


df7 = remove_pps_outliers(df6)
df7.shape


# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

# In[47]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.bhk ==2)]
    bhk3 = df[(df.location == location) & (df.bhk ==3)]
    plt.figure(figsize=(12,8), dpi = 200)
    plt.scatter(bhk2.total_sqft,bhk2.price,color ='blue', label = '2 BHK', s = 50)
    plt.scatter(bhk3.total_sqft,bhk3.price, marker = '+',color = 'green', label = '3 BHK', s = 50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Prie')
    plt.title(location)
    plt.legend()


# In[48]:


plot_scatter_chart(df7,'Rajaji Nagar')


# In[49]:


plot_scatter_chart(df7,'Hebbal')


# **We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.**
# 
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# 
# 
# **Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment**

# In[50]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)


# In[51]:


df8.shape


# In[52]:


plot_scatter_chart(df8,'Hebbal')


# In[53]:


plt.figure(figsize = (20,10),dpi = 200)
plt.hist(df8.price_per_sqft, rwidth = 0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')


# **Outlier Removal Using Bathrooms Feature**

# In[54]:


df8.bath.unique()


# In[55]:


df8[df8.bath>10]


# In[56]:


plt.hist(df8.bath,rwidth =0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('Count')


# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# In[57]:


df8[df8.bath>df8.bhk+2]


# Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

# In[58]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[59]:


df9.head()


# In[60]:


df10 = df9.drop(['size','price_per_sqft'],axis = 1)
df10.head()


# ## Model Building

# In[61]:


dummies = pd.get_dummies(df10.location,drop_first = True)
dummies.head()


# In[62]:


df11 = pd.concat([df10,dummies], axis = 1)


# In[63]:


df11.head()


# In[64]:


df12 = df11.drop('location', axis = 1)


# In[65]:


df12.head()


# In[66]:


df12.shape


# Features and Labels

# In[67]:


X = df12.drop('price',axis = 1)


# In[68]:


X.head()


# In[69]:


y = df12.price


# In[70]:


y.head()


# Train Test split

# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# In[72]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)


# In[73]:


lr_clf.score(X_test,y_test)


# **Use K Fold cross validation to measure accuracy of our LinearRegression model**

# In[74]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSpl`it(n_splits = 5, test_size = 0.2, random_state =0)

cross_val_score(lr_clf,X_train,y_train,cv = cv)


# **We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose**

# **Find best model using GridSearchCV**

# In[75]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# **Based on above results we can say that LinearRegression gives the best score. Hence we will use that.**

# ## Price Prediction

# In[76]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[77]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[78]:


predict_price('1st Phase JP Nagar',1000, 2, 3)


# In[79]:


predict_price('Indira Nagar',1000, 2, 2)


# In[80]:


predict_price('Indira Nagar',1000, 3, 3)


# ## Model deployment

# In[81]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[87]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# ##              THE END
