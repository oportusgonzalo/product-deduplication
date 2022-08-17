
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
country = ''
parent_chain = '' # lower case and "clean"
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
    # dict to map item_name with image_url:
    item_name_image_dict = dict(zip(data['item_name'], data['image_url']))
    data = data.loc[:, ['item_uuid', 'item_name', 'number_sku_sold']]

    return data, item_name_image_dict

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

def raw_vs_clean_name_mapping(df_nlp, item_name_image_dict): 
    print('Saving file to back propagate matches..') 
    df_back_propagation = df_nlp.loc[:, ['item_uuid', 'item_name', 'product_name']]
    # adding image_url column
    df_back_propagation['image_url'] = df_back_propagation['item_name'].map(item_name_image_dict)
    clean_product_image_dict = dict(zip(df_back_propagation['product_name'], df_back_propagation['image_url']))
    df_back_propagation.to_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv', index=False)
    return df_back_propagation, clean_product_image_dict

def pareto_products(data):
    print(f'Identifying the products that represent the 80% of the sales..')

    pareto_df = data.loc[:, ['product_name', 'number_sku_sold']]
    pareto_df = pareto_df.drop_duplicates().reset_index(drop=True)

    # grouping to aggregate units sold
    pareto_df = pareto_df.groupby('product_name').agg({'number_sku_sold': sum}).reset_index()
    pareto_df = pareto_df.sort_values(by='number_sku_sold', ascending=False).reset_index(drop=True)

    # cumulative aggregations to filter 80/20
    pareto_df['cumulate'] = pareto_df["number_sku_sold"].cumsum()
    pareto_df["cum_percentage"] = (pareto_df['cumulate'] / pareto_df["number_sku_sold"].sum()) * 100

    pareto_set = list(set(pareto_df[pareto_df['cum_percentage'] <= 80]['product_name']))
    print(f'Number of products that represent Pareto 80/20: {len(pareto_set)}')
    print(f'Percentage of products that represent Pareto 80/20: {round(len(pareto_set)/len(pareto_df["product_name"].unique()), 3)}')

    return pareto_set

def direct_matches(data_nlp):
    print('Identifying direct matches: member --> canonical_member')

    # reading file with links between raw items and canonical data (when we run bivariate, this file has been already created)
    canonical_links = pd.read_csv('canonical_data/canonical_links.csv')

    for col in ['canonical_leader', 'canonical_member']:
        canonical_links[col] = canonical_links[col].str.lower()
    
    canonical_members = list(set(canonical_links['canonical_member']))

    # dataframe with direct matches
    direct_df = data_nlp[data_nlp['product_name'].isin(canonical_members)].reset_index(drop=True)

    if direct_df.shape[0] > 0:
        direct_members = list(set(direct_df['product_name']))
        print(f'Number of direct matches: {len(direct_members)}')

        # removing products that don't have direct matches
        data_not_direct = data_nlp[~data_nlp['product_name'].isin(direct_members)].reset_index(drop=True)

        # save link between: member --> canonical_member
        direct_df.drop(['item_uuid', 'item_name'], axis=1, inplace=True)
        direct_df = direct_df.merge(canonical_links, how='inner', left_on='product_name', right_on='canonical_member')
        direct_matches_df = direct_df.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member']]
        direct_matches_df = direct_matches_df.drop_duplicates().reset_index(drop=True)

        print(f'Validation - Number of direct matches: {len(direct_matches_df["canonical_member"].unique())}')
        
    else:
        print(f'Number of direct matches: 0')
        # just to keep structure
        data_not_direct = data_nlp.copy()
        direct_matches_df = pd.DataFrame() # --> empty DF (PROBABLY NOT USEFUL TO RETURN IT)

    # return case: no direct matches
    return data_not_direct, canonical_links, direct_matches_df


def product_space_to_detect_similarities(data_not_direct, canonical_links):
    print(f'Preparing set to identify similiarities by TF-IDF + Fuzzy..')

    applicants_not_direct = list(data_not_direct['product_name'].unique())
    # we use leaders as they reduce the space and represent the members
    canonical_leaders = list(canonical_links[~canonical_links['canonical_leader'].isna()]['canonical_leader'].unique())
    # concatenation of: applicants with no direct match + canonical leaders
    product_space = list(set(applicants_not_direct + canonical_leaders))
    print(f'Number of products to match and group: {len(product_space)}')
    return product_space

def leaders_lead(canonical_links, groups_df):
    print(f'Making sure leaders are leaders..')
    canonical_leaders = canonical_links['canonical_leader'].unique()
    # identifying all groups where canonical members are present
    canonical_leaders_group_df = groups_df.loc[groups_df['member'].isin(canonical_leaders)][['group_id', 'member']].drop_duplicates().reset_index(drop=True)
    # dict to replace: group leader by canonical_leader
    canonical_leader_replace_dict = dict(zip(canonical_leaders_group_df['group_id'], canonical_leaders_group_df['member']))
    # adding canonical leader lable: are we able to modify the leader?
    groups_df['modify_leader'] = 'Yes'
    # replace canonical leaders in potential group leader column
    for group_id, leader in canonical_leader_replace_dict.items():
        groups_df.loc[groups_df['group_id'] == group_id, ['leader', 'modify_leader']] = leader, 'No'
    # removing canonical leaders from members --> may lead to issue (leaders being mapped to other leaders)
    groups_df = groups_df[~groups_df['member'].isin(canonical_leaders)].copy()
    return groups_df

def extracting_pareto_groups(groups_df, pareto_set):
    print(f'Extracing groups where pareto members are assigned..')
    
    groups_in_pareto = list(set(groups_df[(groups_df['leader'].isin(pareto_set))|(groups_df['member'].isin(pareto_set))]['group_id']))
    
    pareto_groups_df = groups_df[groups_df['group_id'].isin(groups_in_pareto)].reset_index(drop=True)
    non_pareto_groups_df = groups_df[~groups_df['group_id'].isin(groups_in_pareto)].reset_index(drop=True)

    print(f'Pareto dataframe shape to be reviewed by agents: {pareto_groups_df.shape[0]}')

    return pareto_groups_df, non_pareto_groups_df

def main():
    # Initial time
    t_initial = gets_time()
    # reading CSV files: canonical & applicnats
    data, item_name_image_dict = read_and_select()
    # NLP + regex product name cleaning --> new column: product_name
    data_nlp = nlp_regex_cleaning(language_, data)
    # saving raw product name - clean product name (post NLP + regex): mapping
    df_back_propagation, clean_product_image_dict = raw_vs_clean_name_mapping(data_nlp, item_name_image_dict)
    # Identifying direct matches: member --> canonical_member
    data_not_direct, canonical_links, direct_matches_df = direct_matches(data_nlp)
    # identifies the 20% of the products that represent the 80% of the sales
    pareto_set = pareto_products(data)
    # Preparing set to run grouping script
    product_space = product_space_to_detect_similarities(data_not_direct, canonical_links)
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
    groups_df = leaders_lead(canonical_links, groups_df)
    
    # re-organizing and removing non pareto products
    groups_df = groups_df.sort_values(by=['leader', 'member']).reset_index(drop=True)
    groups_df['image_url'] = groups_df['member'].map(clean_product_image_dict)

    pareto_groups_df, non_pareto_groups_df = extracting_pareto_groups(groups_df, pareto_set)

    # saving results
    pareto_groups_df.to_csv(f'bivariate_outputs/bivariate_pareto_groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)
    non_pareto_groups_df.to_csv(f'bivariate_outputs/bivariate_non_pareto_groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)
    direct_matches_df.to_csv(f'bivariate_outputs/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv', index=False)

    
    # Complete run time
    t_complete = gets_time() - t_initial
    print(f'Time to run the script: {round(t_complete/60, 3)} minutes!')
    print('Success!')

    return pareto_groups_df, non_pareto_groups_df, df_back_propagation

if __name__ == "__main__":
    main()











