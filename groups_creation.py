
import pandas as pd
import numpy as np
import re
import time
import sys
from fuzzywuzzy import fuzz
import logging

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


# DEFINITIONS

# file to work on
csv_file = 'uk_booker_products'
# parameters (MUST DO! -- are used to specify the outputs name)
country = 'uk'
parent_chain_use = True # will be false when the complete set corresponds to a specific chain (also: no chain name variations)
parent_chain = 'booker' # lower case and "clean"
parent_chain_column = 'parent_chain_name'
item_column = 'item_name'
language_ = 'en'

# hiperparameters
threshold_products = 85
threshold_package = 75


def read_and_select(csv_file):
    logging.info('Reading file and selecting data to work on..')
    # reading raw data
    data = pd.read_csv(f'data/{csv_file}.csv')

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

    return df_nlp

def nlp_regex_cleaning(language_, df_nlp):

    if language_ == 'en':
        stop_words = stopwords.words('english')
    elif language_ == 'es':
        stop_words = stopwords.words('spanish')

    regex_clean = r'(pm \d+\w+)|(pm \d+\.\d+)|(pm\d+\.\d+)|(\d+ pmp)|(pm\d+)|( \.+)|(pmp\d+.\d+)|(\d+pmp)|(pmp \d+)|(\d+.\d+ pm)'
    df_nlp = nlp_cleaning(df_nlp, stop_words, regex_clean)

    print(f'Percentage of unique products after NLP: {round(len(df_nlp.product_name.unique())/len(df_nlp.item_name.unique()), 3)}')

    return df_nlp

def raw_vs_clean_name_mapping(df_nlp):   
    df_back_propagation = df_nlp.loc[:, ['item_name', 'product_name']]
    df_back_propagation.to_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv', index=False)
    return df_back_propagation


def tf_idf_method(df_nlp):
    # preparing set for TF-IDF
    df_tf = df_nlp.loc[:, ['product_name']]
    df_tf = df_tf.drop_duplicates().reset_index(drop=True)
    df_tf['id'] = range(1, len(df_tf) + 1)

    # create object
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2, token_pattern='(\S+)')
    # get tf-idf values
    tf_idf_matrix = tfidf_vectorizer.fit_transform(df_tf['product_name'])

    return df_tf, tf_idf_matrix

def cosine_similarity_calculation(df_tf, tf_idf_matrix):

    matches = cosine_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), 25, 0)
    
    # ### Create a match table to show the similarity scores
    matches_df = pd.DataFrame()
    matches_df = get_matches_df(matches, df_tf['product_name'], top=False)
    matches_df = matches_df.drop_duplicates().reset_index(drop=True)

    return matches_df

def fuzzy_ratios(matches_df):
    print(f'Product Threshold: {threshold_products}')
    # Fuzzy ratios calculation
    matches_df['fuzz_ratio'] = matches_df.apply(lambda x: fuzz.token_sort_ratio(x['product_name'], x['match']), axis=1)

    #  Keeping products with high similarity
    df_similars = matches_df[matches_df['fuzz_ratio'] >= threshold_products].\
                        drop_duplicates(subset=['product_name', 'match']).reset_index(drop=True)

    df_similars = df_similars.sort_values(by=['product_name', 'match']).reset_index(drop=True)

    return df_similars

def extends_similarities(df_similars):
    # copy of dataframe
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
    
    return df_similars_ext

def cleaning_by_package_similarity(df_similars_ext):
    reg_package = r'(\d+x\d+\w+)|(\d+ x \d+\w+)|(\d+\.+\d+\w+)|(\d+\.+\d+ \w+)|(\d+ ml)|(\d+ g)|(\d+\w+)|(\d+ \w+)'
    # extracting package
    df_similars_ext['package'] = package_extract(df_similars_ext, 'product_name', reg_package)
    df_similars_ext['package_candidate'] = package_extract(df_similars_ext, 'candidate', reg_package)
    # package similarity
    df_similars_ext['package_ratio'] = df_similars_ext.apply(lambda x: fuzz.token_sort_ratio(x['package'],\
                                                                                x['package_candidate']), axis=1)
                                                                                
    # Package filter + Column selection
    print(f'Package Threshold: {threshold_package}')
    df_clean = df_similars_ext[df_similars_ext['package_ratio'] > threshold_package].reset_index(drop=True)
    df_clean = df_clean.loc[:, ['product_name', 'candidate']]
    
    return df_clean

def product_name_replacement(df, dic_):
    df['product_name'] = df['product_name'].map(dic_)
    df['candidate'] = df['candidate'].map(dic_)
    return df

def creating_product_index_name_mapping_dict(df_tf):
    # To  tansform product names into integers (easier to compare)
    product_index_dict = dict(zip(df_tf['product_name'], df_tf.index))
    index_product_dict = dict(zip(df_tf.index, df_tf['product_name']))
    return product_index_dict, index_product_dict

def groups_concatenation(df_clean, df_similars, index_product_dict):
    # list of products
    clean_leaders = df_clean['product_name'].unique()
    print(f'Leaders: {len(clean_leaders)}; Similar products: {len(df_similars["match"].unique())}')

    # time before
    t_bef_group = gets_time()

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
    t_run = gets_time() - t_bef_group
    print(f'Time to run group concatenation: {round(t_run/60, 3)} minutes!')

    return groups_df, track_df

def main():
    # Initial time
    t_initial = gets_time()
    # reading CSV file, cleaning parent_chain_name, and selecting data to work on
    df_nlp = read_and_select(csv_file)
    # NLP + regex product name cleaning --> new column: product_name
    df_nlp = nlp_regex_cleaning(language_, df_nlp)
    # saving raw product name - clean product name (post NLP + regex) mapping
    df_back_propagation = raw_vs_clean_name_mapping(df_nlp)
    # Appying TF-IDF method
    df_tf, tf_idf_matrix = tf_idf_method(df_nlp)
    # Applying cosine similarity to detect most similar products (potential group)
    matches_df = cosine_similarity_calculation(df_tf, tf_idf_matrix)
    # Calculating fuzzy ratios and keeping products with similarity above threshold_products
    df_similars = fuzzy_ratios(matches_df)
    # extending product similarities: A similar to B, and B similar to D; then A, B, and D are similars
    df_similars_ext = extends_similarities(df_similars)
    # calculating fuzzy ratios between product packages, keeping similarities above threshold_package
    df_clean = cleaning_by_package_similarity(df_similars_ext)
    # dictionaries to map product_name --> index
    product_index_dict, index_product_dict = creating_product_index_name_mapping_dict(df_tf)
    # product names into integers --> easy to compare
    df_clean = product_name_replacement(df_clean, product_index_dict)
    # concatenating groups to global dataframe
    groups_df, track_df = groups_concatenation(df_clean, df_similars, index_product_dict)

    # Saving results
    groups_df = groups_df.sort_values(by=['leader', 'member']).reset_index(drop=True)
    groups_df.to_csv(f'outputs/groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)

    # Complete run time
    t_complete = gets_time() - t_initial
    print(f'Time to run the script: {round(t_complete/60, 3)} minutes!')

    return groups_df, track_df, df_back_propagation

if __name__ == "__main__":
    main()











