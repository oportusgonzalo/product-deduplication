
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

# parameters (MUST DO! -- are used to specify the outputs name)
country = 'uk'
parent_chain_use = False # will be false when the complete set corresponds to a specific chain (also: no chain name variations)
parent_chain = 'nisa' # lower case and "clean"
parent_chain_column = 'parent_chain_name'
item_column = 'item_name'
language_ = 'en'

# canonical file
canonical_file = 'canonical_catalog'

# hiperparameters
threshold_products = 85
threshold_package = 75


def read_and_select():
    print('Reading canonical and applicants files..')
    
    data = pd.read_csv(f'data/{country}_{parent_chain}_uuid_name.csv')
    data = data.loc[:, ['item_uuid', 'item_name']]
    canonical_df = pd.read_csv(f'canonical_data/{canonical_file}.csv')
    canonical_df.rename(columns={'group_id': 'canonical_id', 'leader': 'canonical_leader', 'member': 'canonical_member'}, inplace=True)

    return data, canonical_df

def nlp_regex_cleaning(language_, data):
    print('NLP + Regex product name cleaning..')

    if language_ == 'en':
        stop_words = stopwords.words('english')
    elif language_ == 'es':
        stop_words = stopwords.words('spanish')

    regex_clean = r'(pm \d+\w+)|(pm \d+\.\d+)|(pm\d+\.\d+)|(\d+ pmp)|(pm\d+)|( \.+)|(pmp\d+.\d+)|(\d+pmp)|(pmp \d+)|(\d+.\d+ pm)'
    data_nlp = nlp_cleaning(data, stop_words, regex_clean)

    print(f'Percentage of unique products after NLP: {round(len(data_nlp.product_name.unique())/len(data_nlp.item_name.unique()), 3)}')

    return data_nlp.loc[:, ['item_uuid', 'item_name', 'product_name']]

def direct_matches(data_nlp, canonical_df):
    print('Identifying direct matches: member --> canonical_member')
    
    canonical_members = list(set(canonical_df['canonical_member']))

    # dataframe with direct matches
    direct_df = data_nlp[data_nlp['product_name'].isin(canonical_members)].reset_index(drop=True)

    if direct_df.shape[0] > 0:
        direct_members = list(set(direct_df['product_name']))
        print(f'Number of existing matches: {len(direct_members)}')

        # removing products that don't have direct matches
        data_not_direct = data_nlp[~data_nlp['product_name'].isin(direct_members)].reset_index(drop=True)

        # save link between: member --> canonical_member
        direct_df = direct_df.merge(canonical_df, how='inner', left_on='product_name', right_on='canonical_member')
        direct_matches_df = direct_df.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member']]
        
        # return case: we have direct matches
        return data_not_direct, canonical_df, direct_matches_df
    else:
        print(f'Number of existing matches: 0')
        # just to keep structrue
        data_not_direct = data_nlp.copy()
        direct_matches_df = pd.DataFrame() # --> empty DF (PROBABLY NOT USEFUL TO RETURN IT)

        # return case: no direct matches
        return data_not_direct, canonical_df, direct_matches_df


def product_space_to_detect_similarities(data_not_direct, canonical_df):
    print(f'Preparing set to identify similiarities by TF-IDF + Fuzzy..')

    applicants_not_direct = list(data_not_direct['product_name'].unique())
    # we use leaders as they reduce the space and represent the members
    canonical_leaders = list(canonical_df['canonical_leader'].unique())
    # concatenation of: applicants with no direct match + canonical leaders
    product_space = list(set(applicants_not_direct + canonical_leaders))
    print(f'Number of products in the set: {len(product_space)}')
    return product_space

def raw_vs_clean_name_mapping(df_nlp): 
    print('Saving file to back propagate matches..')  
    df_back_propagation = df_nlp.loc[:, ['item_name', 'product_name']]
    df_back_propagation.to_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv', index=False)
    return df_back_propagation

def leaders_lead(canonical_df, groups_df):
    print(f'Making sure leaders are leaders..')
    canonical_leaders = canonical_df['canonical_leader'].unique()
    # identifying all groups where canonical members are present
    canonical_leaders_group_df = groups_df.loc[groups_df['member'].isin(canonical_leaders)][['group_id', 'member']].drop_duplicates().reset_index(drop=True)
    # dict to replace: group leader by canonical_leader
    canonical_leader_replace_dict = dict(zip(canonical_leaders_group_df['group_id'], canonical_leaders_group_df['member']))
    # replace canonical leaders in potential group leader column
    for group_id, leader in canonical_leader_replace_dict.items():
        groups_df.loc[groups_df['group_id'] == group_id, 'leader'] = leader
    return groups_df


def main():
    # Initial time
    t_initial = gets_time()
    # reading CSV files: canonical & applicnats
    data, canonical_df = read_and_select()
    # NLP + regex product name cleaning --> new column: product_name
    data_nlp = nlp_regex_cleaning(language_, data)
    # Identifying direct matches: member --> canonical_member
    data_not_direct, canonical_df, direct_matches_df = direct_matches(data_nlp, canonical_df)
    # Preparing set to run grouping script
    product_space = product_space_to_detect_similarities(data_not_direct, canonical_df)
    df_product_space = pd.DataFrame(data={'product_name': product_space})

    # Appying TF-IDF method
    df_tf, tf_idf_matrix = tf_idf_method(df_product_space)
    # Applying cosine similarity to detect most similar products (potential group)
    matches_df = cosine_similarity_calculation(df_tf, tf_idf_matrix)
    # Calculating fuzzy ratios and keeping products with similarity above threshold_products
    df_similars = fuzzy_ratios(matches_df, threshold_products)
    # extending product similarities: A similar to B, and B similar to D; then A, B, and D are similars
    df_similars_ext = extends_similarities(df_similars)
    # calculating fuzzy ratios between product packages, keeping similarities above threshold_package
    df_clean = cleaning_by_package_similarity(df_similars_ext, threshold_package)
    # dictionaries to map product_name --> index
    product_index_dict, index_product_dict = creating_product_index_name_mapping_dict(df_tf)
    # product names into integers --> easy to compare
    df_clean = product_name_replacement(df_clean, product_index_dict)
    # concatenating groups to global dataframe
    groups_df, track_df = groups_concatenation(df_clean, df_similars, index_product_dict)

    # leaders lead
    groups_df = leaders_lead(canonical_df, groups_df)
    
    # saving results
    groups_df.to_csv(f'bivariate_outputs/bivariate_groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)
    direct_matches_df.to_csv(f'bivariate_outputs/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)


    # Complete run time
    t_complete = gets_time() - t_initial
    print(f'Time to run the script: {round(t_complete/60, 3)} minutes!')
    print('Success!')

    #return groups_df, track_df, df_back_propagation

if __name__ == "__main__":
    main()











