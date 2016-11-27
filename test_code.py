
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import gzip
import time


# In[2]:

# ==== Reading data ====== #
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


# Tools and Home Improvement	reviews (1,926,047 reviews) metadata (269,120 products)

data_review = getDF('reviews_Tools_and_Home_Improvement_5.json.gz')
data_meta = getDF('meta_Tools_and_Home_Improvement.json.gz')


# In[3]:

data_review.head(n=1) # preview of the reviews data


# In[4]:

data_meta.head(n=1) # preview of the items data
data_meta_upd = data_meta.set_index('asin')


# Item-to-item CF matches item purchased or rated by a target user to similar items and combines those similar items in a recommendation list.
# 
# Similarity can be computed in a number of ways
#     - Using the user ratings
#     - Using some product description
#     - Using co-occurrence of items in a bag or in the set of a user past purchased products 
#     
# Generating the prediction: look into the target userʼs ratings and use techniques to obtain predictions based on the ratings of similar products 
# 
# 1. Scan the products, and for all the customers that bought a product, identify the other products bought by those customers
# 
# 2. Then compute the similarity only for these pairs 
# 
# 

# In[5]:

# lists of the all users IDs and all product IDs (unique)
users_ids = list(set(data_review['reviewerID']))
products_asins = list(set(data_review['asin']))


# In[92]:

# grouping of data by users and by products
s_data_review = data_review[['reviewerID','asin','overall']]
reviews_grouped_by_product = s_data_review.groupby('asin')
reviews_grouped_by_user = s_data_review.groupby('reviewerID')


# In[7]:

def who_also_bought(product_id):
    # input - product ID, output - dataframe with data = IDs of users, who also bought this product
    return reviews_grouped_by_product.get_group(product_id)['reviewerID']

def get_product_list(customer_id):
    # input - user ID, output - dataframe with data = what products was rated by the given user
    return reviews_grouped_by_user.get_group(customer_id)['asin']

def do_df(asin):
    # input - product ID, output - dataframe with data = ratings for this product from the users who have rated it
    tmp = reviews_grouped_by_product.get_group(asin)[['reviewerID','overall']]
    tmp.set_index('reviewerID', inplace = True)
    tmp.columns = [asin]
    del tmp.index.name
    return tmp


# In[165]:

# list of dataframes; 
#each dataframe is with: index = users ID, column = product ID, data = rating for product from users
reviews_matrix = [] 

# dictionary of dictionaries; 
# format: item_X = {item_A: # of times user rated item_A AND item_X}
purchase_frequency = {}


# In[192]:

example_id = 'AHQRU3MRORIWQ'
example_user_catalog = reviews_grouped_by_user.get_group(example_id)['asin']


# In[201]:

start_time = time.time()

cutomers_already_covered = [example_id]
products_already_covered = list(example_user_catalog.values)

#TODO: find a better way than nested for loops

for item_id in example_user_catalog:
    customers = who_also_bought(item_id)
    cutomers_already_covered.append(list(customers.values))
    purchase_frequency[item_id] = {}
    
    for c_id in customers:
        if np.all(c_id not in cutomers_already_covered):
            products = get_product_list(c_id)
            products_already_covered.append(list(products.values))
            #products = products[products != item_id] 
            
            for prod in products:
                if np.all(prod not in products_already_covered):
                    reviews_matrix.append( do_df(prod) )
                    try: 
                        purchase_frequency[item_id][prod] = purchase_frequency[item_id][prod] + 1
                    except KeyError:     
                        purchase_frequency[item_id][prod] = 1
            
print("--- %s seconds ---" % (time.time() - start_time))  

start_time = time.time()

matrix = pd.concat(reviews_matrix, axis = 1, join='outer')
#matrix.fillna(0, inplace = True)
#df_purchase_frequency = pd.DataFrame(purchase_frequency)

sorted_purchase_frequency = {}
for val in purchase_frequency.keys():
    sorted_purchase_frequency[val] = sorted(purchase_frequency[val], key=purchase_frequency[val].__getitem__, reverse=True)

print("--- %s seconds ---" % (time.time() - start_time))  


# In[177]:

matrix.head()


# Matrix contains the reviewes of the users for the all related products
# 
# Sorted_purchase_frequency containes for each item in the users catalogue the information how often the product is bought together (the most frequent in the beginning)
# 
# The next steps:
# 
# 1 . define the top N (to be set) products based on the how often bought together

# In[54]:

topN = 10 # arbitrary set
top_items_people_also_bought = (df_purchase_frequency.sum(axis = 1)).sort_values(ascending = False)
topN_products = top_items_people_also_bought[:topN].index.values


# In[188]:

topN = 10
topN_products = []
for val in sorted_purchase_frequency.keys():
    topN_products = topN_products + (sorted_purchase_frequency[val][1:topN])

# TODO: the information that the product is a duplicate can be usefull, need to include into analysis
# delete the duplicates
topN_products = list(set(topN_products))


# 2 . In top N define M most similar products to ones that are in the user basket
# 
# To define it, need to look at the products similarity. The possible features:
# - average rating
# - price
# - category
# - brand

# In[352]:

# User Products Catalogue

user_products_data = data_meta_upd.loc[example_user_catalog,['categories','price','brand','salesRank']]
for el in example_user_catalog:
    user_products_data.loc[el,'Rating'] = reviews_grouped_by_product.get_group(el)['overall'].mean()
    try:
        user_products_data.loc[el,'SalesCategory'] = user_products_data.loc[el, 'salesRank'].keys()
        k = list(user_products_data.loc[el,'SalesCategory'])
        user_products_data.loc[el,'SalesRank'] = user_products_data.loc[el, 'salesRank'][k[0]]
    except AttributeError:
        user_products_data.loc[el,'SalesCategory'] = np.nan
        user_products_data.loc[el,'SalesRank'] = np.nan
        
user_products_data = user_products_data.loc[:,['Rating','price','SalesCategory','SalesRank','brand', 'categories']]
#user_products_data = user_products_data.loc[:,['price','SalesCategory','SalesRank','brand', 'categories']]
del user_products_data.index.name
# treat NA as 0s
user_products_data.fillna(0, inplace = True)


# In[353]:

user_products_data.head(n=2)


# In[293]:

user_products_data.shape


# In[355]:

# Top N Products Catalogue
topN_products_data = data_meta_upd.loc[pd.Series(topN_products),['categories','price','brand','salesRank']]
for el in topN_products:
    topN_products_data.loc[el,'Rating'] = reviews_grouped_by_product.get_group(el)['overall'].mean()
    try:
        topN_products_data.loc[el,'SalesCategory'] = topN_products_data.loc[el, 'salesRank'].keys()
        k = list(topN_products_data.loc[el,'SalesCategory'])
        topN_products_data.loc[el,'SalesRank'] = topN_products_data.loc[el, 'salesRank'][k[0]]
    except AttributeError:
        topN_products_data.loc[el,'SalesCategory'] = np.nan
        topN_products_data.loc[el,'SalesRank'] = np.nan
        
topN_products_data = topN_products_data.loc[:,['Rating','price','SalesCategory','SalesRank','brand', 'categories']]
#topN_products_data = topN_products_data.loc[:,['price','SalesCategory','SalesRank','brand', 'categories']]

del topN_products_data.index.name

# treat NA as 0s
topN_products_data.fillna(0, inplace = True)


# In[359]:

topN_products_data.head(n=3)


# Each item is characterized by the vector (average rating, price, sales category, sales rank, brand, categories)
# 
# #TODO: improve
# 
# To find the similarity:
# 
# For the numeric variables - rating and price:
# - how close are ratings 
# - what price range the user is focused on (based on the mean of the items prices he has bought) 
# 
# For the categorical variables - sales category, brand, categories:
# - sales category and brand: 1 if coincide, 0 if not
# - categories: measure the overlap 
# 

# In[389]:

def similarity_rate(product1, product2):
    # TODO: implement other measures for the categorical variables of SalesCategory, SalesRank, brand, categories_overlap
    # for the categorical values the weighted average overlap measure is applied 
    
    #product1 = topN_products_data.loc[asin1,:]
    #product2 = topN_products_data.loc[asin2,:]
    
    # numeric meaures
    rating_similarity = np.exp(-abs(product1['Rating'] - product2['Rating'])) #closer - higher similarity
    price_similarity = np.exp(-abs(product1['price'] - product2['price'])) #closer - higher similarity
    
    # overlap meaures
    brand_similarity = product1['brand'] == product2['brand']
    
    sales_category_similarity = product1['SalesCategory'] == product2['SalesCategory']
    sales_rank_similarity = np.exp(-abs(product1['SalesRank'] - product2['SalesRank']))
    sales_similarity = (sales_category_similarity * sales_rank_similarity) if sales_category_similarity!=0 else 0.
    
    #Category similarity defines as a % out of the max possible similarity
    max_similarity = max(len(product2['categories'][0]), len(product1['categories'][0]))
    category_similarity = len(set(product2['categories'][0]).intersection(product1['categories'][0]))/ max_similarity
    
    return np.mean([rating_similarity, price_similarity, brand_similarity, sales_similarity, category_similarity])
    


# In[399]:

similarity_matrix = pd.DataFrame([], index = topN_products_data.index.values, columns = user_products_data.index.values)

start_time = time.time()
for topN_el in similarity_matrix.index.values:
    p2 = topN_products_data.loc[topN_el,:]
    for user_prod in similarity_matrix.columns.values:
        p1 = user_products_data.loc[user_prod,:]
        similarity_matrix.loc[topN_el,user_prod] = similarity_rate(p1,p2)
        
print("--- %s seconds ---" % (time.time() - start_time))          


# In[402]:

similarity_matrix.head(n = 2)


# In[411]:

sorted_average_similarity = (similarity_matrix.mean(axis = 1)).sort_values(ascending = False)
# select top 5 to recomment

prod_IDs_to_recommend = sorted_average_similarity[1:6]
prod_IDs_to_recommend


# In[410]:

topN_products_data.loc[prod_IDs_to_recommend.index.values,:]


# In[412]:

# original user dataset
user_products_data


# In[ ]:



