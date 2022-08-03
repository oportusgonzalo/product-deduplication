
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
import time
import sys

from static import *

# To calculate: TF-IDF & Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

import warnings
warnings.filterwarnings("ignore")


import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords


def gets_time():
    return time.time()

# Initial time
t_initial = gets_time()


# ## 1. Pre-processing

# parameters
country = 'uk'
parent_chain = 'booker' # lower case and "clean"
parent_chain_column = 'parent_chain_name'
parent_chain_use = True
item_column = 'item_name'
language_ = 'en'

# hiperparameters
threshold_products = 85
threshold_package = 75

# reading raw data
data = pd.read_csv('data/uk_booker_products.csv')

if parent_chain_use:
    # cleaning parent chain name as it has duplicated entries
    df = clean_text(data, parent_chain_column, '{}_{}'.format(parent_chain_column, 'norm'))
    # chain selection and columns to work on
    df_nlp = df[df['parent_chain_name_norm'] == parent_chain]
    df_nlp = df_nlp.loc[:, ['parent_chain_name_norm', item_column]].reset_index(drop=True)
else:
    df_nlp = data.loc[:, [item_column]].drop_duplicates().reset_index(drop=True)

# item name standardization
df_nlp.rename(columns={'sku_name': 'item_name'}, inplace=True)

print(f"Initial products: {len(list(set(df_nlp['item_name'].unique())))}")

if language_ == 'en':
    stop_words = stopwords.words('english')
elif language_ == 'es':
    stop_words = stopwords.words('spanish')

regex_clean = r'(pm \d+\w+)|(pm \d+\.\d+)|(pm\d+\.\d+)|(\d+ pmp)|(pm\d+)|( \.+)|(pmp\d+.\d+)|(\d+pmp)|(pmp \d+)|(\d+.\d+ pm)'

df_nlp = nlp_cleaning(df_nlp, stop_words, regex_clean)


print(f'Percentage of unique products after NLP: {round(len(df_nlp.product_name.unique())/len(df_nlp.item_name.unique()), 3)}')

df_back_propagation = df_nlp.loc[:, ['item_name', 'product_name']]

df_back_propagation.to_csv(f'back_propagation/groups_{country}_back_propagation.csv', index=False)


# ## 3. TF-IDF Application

# ### Creating a tf-idf matrix

# preparing set for TF-IDF
df_tf = df_nlp.loc[:, ['product_name']]
df_tf = df_tf.drop_duplicates().reset_index(drop=True)
df_tf['id'] = range(1, len(df_tf) + 1)

# ### Applying method

# create object
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2, token_pattern='(\S+)')

# get tf-idf values
tf_idf_matrix = tfidf_vectorizer.fit_transform(df_tf['product_name'])

# ## 4. Computing cosine similarity

matches = cosine_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), 25, 0)

# ### Create a match table to show the similarity scores
matches_df = pd.DataFrame()
matches_df = get_matches_df(matches, df_tf['product_name'], top=False)

matches_df = matches_df.drop_duplicates().reset_index(drop=True)

# ## 5. Grouping products

# ### 5.1 Fuzzy ratios calculation
matches_df['fuzz_ratio'] = matches_df.apply(lambda x: fuzz.token_sort_ratio(x['product_name'], x['match']), axis=1)

# ### 5.2 Keeping products with high similarity
print(f'Product Threshold: {threshold_products}')

df_similars = matches_df[matches_df['fuzz_ratio'] >= threshold_products].\
                        drop_duplicates(subset=['product_name', 'match']).reset_index(drop=True)

# ### 5.3 Logic to aggregate
df_similars = df_similars.sort_values(by=['product_name', 'match']).reset_index(drop=True)


# ### a) Extending similarities
df_similars_copy = df_similars.drop(columns=['similarity_score', 'fuzz_ratio'], axis=1).copy()
df_similars_copy.rename(columns={'match': 'extended_match', 'product_name': 'match'}, inplace=True)

# extending
df_similars_mrg = df_similars.merge(df_similars_copy, how='inner', on='match')
df_similars_mrg.drop('similarity_score', axis=1, inplace=True)

# melt dataframe
df_melt = df_similars_mrg.melt(id_vars=['product_name', 'fuzz_ratio'], var_name='which_match', value_name='candidate')
df_melt = df_melt.drop('which_match', axis=1)[['product_name', 'candidate', 'fuzz_ratio']]

df_similars_ext = df_melt.drop_duplicates(['product_name', 'candidate']).sort_values(by=['product_name', 'candidate'])\
            .reset_index(drop=True)

# ### b) Package similarity
reg_package = r'(\d+x\d+\w+)|(\d+ x \d+\w+)|(\d+\.+\d+\w+)|(\d+\.+\d+ \w+)|(\d+ ml)|(\d+ g)|(\d+\w+)|(\d+ \w+)'
# extracting package
df_similars_ext['package'] = package_extract(df_similars_ext, 'product_name', reg_package)
df_similars_ext['package_candidate'] = package_extract(df_similars_ext, 'candidate', reg_package)
# package similarity
df_similars_ext['package_ratio'] = df_similars_ext.apply(lambda x: fuzz.token_sort_ratio(x['package'],\
                                                                                x['package_candidate']), axis=1)

# ### c) Tansforming product names into integers (easier to compare)
product_index_dict = dict(zip(df_tf['product_name'], df_tf.index))
index_product_dict = dict(zip(df_tf.index, df_tf['product_name']))

for col in ['product_name', 'candidate']:
    df_similars_ext[col] = df_similars_ext[col].map(product_index_dict)

# ### d) Package filter + Column selection
print(f'Package Threshold: {threshold_package}')
df_clean = df_similars_ext[df_similars_ext['package_ratio'] > threshold_package].reset_index(drop=True)
df_clean = df_clean.loc[:, ['product_name', 'candidate']]

# ### e ) Functions

def product_name_replacement(df, dic_):
    df['product_name'] = df['product_name'].map(dic_)
    df['candidate'] = df['candidate'].map(dic_)
    return df


# ### f) Procedure: for each product
clean_leaders = df_clean['product_name'].unique()
print(len(clean_leaders), len(df_similars['match'].unique()))

# time before
t_bef_group = time.time()

# dataframe definition
groups_df = pd.DataFrame(columns=['group_id', 'leader', 'member'])
track_df = pd.DataFrame(columns=['group_id', 'member'])

for leader in clean_leaders:
    select_df = df_clean[df_clean['product_name'] == leader] 
    applicants_list = list(pd.unique(select_df[['product_name', 'candidate']].values.ravel('K')))
    groups_df, track_df = verify_and_concat_groups(groups_df, track_df, leader, applicants_list)


# replacing product names
groups_df['leader'] = groups_df['leader'].map(index_product_dict)
groups_df['member'] = groups_df['member'].map(index_product_dict)


# time run
t_run = time.time()-t_bef_group
print(f'Time to run procedure: {round(t_run/60, 3)} minutes!')

# Saving results
groups_df = groups_df.sort_values(by=['leader', 'member']).reset_index(drop=True)
groups_df.to_csv(f'outputs/groups_{country}_{threshold_products}_{threshold_package}.csv', index=False)

# Complete run time
t_complete = time.time()-t_initial
print(f'Time to run it all: {round(t_complete/60, 3)} minutes!')











