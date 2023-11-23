
import pandas as pd
from fuzzywuzzy import fuzz

from static import *

# To calculate: TF-IDF & Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords


# DEFINITIONS

# parameters
country = ''
parent_chain = ''
item_column = 'item_name'
language_ = 'en'

'''
- Adjust to remove duplicates with PLUs in both sides
'''

# hiperparameters
threshold_products = 95
threshold_package = 95


def read_and_select():
    print('Reading file and selecting data to work on..')
    
    # reading raw data
    data = pd.read_csv(f'data/{country}/{country}_{parent_chain}_uuid_name.csv')
    data = data.replace(r'\N', None)

    # removing the package from the product name
    data['package'] = data['item_name'].str.split('(').str[1].str.split(')').str[0]
    data['raw_item_name'] = data.loc[:, ['item_name']]
    data['item_name'] = data['item_name'].str.split('(').str[0]

    # dict to map item_uuid with image_url:
    item_uuid_image_dict = dict(zip(data['item_uuid'], data['image_url']))

    print(f'Initial dataframe shape: {data.shape}')
    print(f'Initial unique products - messy: {len(list(set(data["item_name"])))}')

    return data, item_uuid_image_dict

def nlp_regex_cleaning(language_, data):
    print('NLP + Regex product name cleaning..')

    if language_ == 'en':
        stop_words = stopwords.words('english')
    elif language_ == 'es':
        stop_words = stopwords.words('spanish')
    elif language_ == 'pt':
        stop_words = stopwords.words('portuguese')

    df_nlp = nlp_cleaning(data, stop_words, regex_clean=False)

    print(f'Percentage of unique products after NLP: {round(len(df_nlp.product_name.unique())/len(df_nlp.item_name.unique()), 3)}')
    print(f'# of unique products after NLP: {len(df_nlp.product_name.unique())}')

    # saving relation between product name & package for later
    clean_name_to_package_dict = dict(zip(df_nlp['product_name'], df_nlp['package']))
    clean_name_to_uuid_dict = dict(zip(df_nlp['product_name'], df_nlp['item_uuid']))
    clean_name_to_raw_name_dict = dict(zip(df_nlp['product_name'], df_nlp['raw_item_name']))

    return df_nlp.loc[:, ['item_uuid', 'raw_item_name', 'item_name', 'product_name', 'package']], clean_name_to_package_dict, clean_name_to_uuid_dict, clean_name_to_raw_name_dict

def tf_idf_method(df_nlp):
    print('Applying TF-IDF method..')
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
    print('Calculating Cosine Similarities..')

    matches = cosine_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), 25, 0)
    
    # create a match table to show the similarity scores
    matches_df = pd.DataFrame()
    matches_df = get_matches_df(matches, df_tf['product_name'], top=False)
    matches_df = matches_df.drop_duplicates().reset_index(drop=True)

    # removing similarity scores equal to 1: same product
    matches_df = matches_df[matches_df['similarity_score'] != 1].reset_index(drop=True)

    return matches_df

def fuzzy_ratios(matches_df, clean_name_to_uuid_dict):
    print('Fuzzy ratios calculation..')
    print(f'Product Threshold: {threshold_products}')

    # removing a fuzzy comparison between the same product
    matches_df = matches_df[matches_df['product_name'] != matches_df['match']].reset_index(drop=True)

    # Fuzzy ratios calculation
    matches_df['fuzz_ratio'] = matches_df.apply(lambda x: fuzz.token_sort_ratio(x['product_name'], x['match']), axis=1)

    #  Keeping products with high similarity
    df_similars = matches_df[matches_df['fuzz_ratio'] >= threshold_products].\
                        drop_duplicates(subset=['product_name', 'match']).reset_index(drop=True)

    df_similars = df_similars.sort_values(by=['product_name', 'match']).reset_index(drop=True)

    return df_similars

def extends_similarities(df_similars):
    print('Extending product similarities..')
    # copy of dataframe
    df_similars_copy = df_similars.drop(columns=['similarity_score', 'fuzz_ratio'], axis=1).copy()
    df_similars_copy.rename(columns={'match': 'extended_match', 'product_name': 'match'}, inplace=True)

    # extending
    df_similars_mrg = df_similars.merge(df_similars_copy, how='left', on='match')
    df_similars_mrg.drop('similarity_score', axis=1, inplace=True)

    # melt dataframe
    df_melt = df_similars_mrg.melt(id_vars=['product_name', 'fuzz_ratio'], var_name='which_match', value_name='candidate')
    df_melt = df_melt.drop('which_match', axis=1)[['product_name', 'candidate', 'fuzz_ratio']]
    df_melt = df_melt[~df_melt['candidate'].isna()].reset_index(drop=True)

    df_similars_ext = df_melt.drop_duplicates(['product_name', 'candidate']).sort_values(by=['product_name', 'candidate'])\
                .reset_index(drop=True)
    
    # removing same product comparisons
    df_similars_ext = df_similars_ext[df_similars_ext['product_name'] != df_similars_ext['candidate']].reset_index(drop=True)

    return df_similars_ext

def cleaning_by_package_similarity(df_similars_ext, clean_name_to_package_dict):
    print('Filtering product matches by package fuzzy ratio similarity measure..')
    
    # adding the package
    df_similars_ext['package'] = df_similars_ext['product_name'].map(clean_name_to_package_dict)
    df_similars_ext['package_candidate'] = df_similars_ext['candidate'].map(clean_name_to_package_dict)

    # package similarity
    df_similars_ext['package_ratio'] = df_similars_ext.apply(lambda x: fuzz.token_sort_ratio(x['package'],\
                                                                                x['package_candidate']), axis=1)
                                                                                
    # Package filter + Column selection
    print(f'Package Threshold: {threshold_package}')
    df_similars_ext = df_similars_ext[df_similars_ext['package_ratio'] > threshold_package].reset_index(drop=True)
    df_clean = df_similars_ext[df_similars_ext['package_ratio'] > threshold_package].reset_index(drop=True)
    df_clean = df_clean.loc[:, ['product_name', 'candidate']]

    print(f'# possible duplicated products: {df_clean.shape[0]}')

    # preserving the similarity thresholds
    df_thresholds = df_similars_ext.loc[:, ['product_name', 'candidate', 'fuzz_ratio', 'package_ratio']]

    return df_clean, df_thresholds

def creating_product_index_name_mapping_dict(df_tf):
    print('Transforming product names into integers for computational purposes..')
    # To  tansform product names into integers (easier to compare)
    product_index_dict = dict(zip(df_tf['product_name'], df_tf.index))
    index_product_dict = dict(zip(df_tf.index, df_tf['product_name']))
    return product_index_dict, index_product_dict

def product_name_replacement(df, dic_):
    df['product_name'] = df['product_name'].map(dic_)
    df['candidate'] = df['candidate'].map(dic_)
    return df

def groups_concatenation(df_clean, index_product_dict):
    print('Concatenating groups to global DF..')

    # list of products
    clean_leaders = list(set(df_clean["product_name"]))
    print(f'Leaders: {len(clean_leaders)}; Similar products: {len(list(set(df_clean["candidate"])))}')

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

    # removing relations to the same product
    groups_df = groups_df[groups_df['leader'] != groups_df['member']].reset_index(drop=True)
    groups_df = groups_df.loc[:, ['leader', 'member']].rename(columns={'leader': 'winner', 'member': 'loser'})

    # time run
    t_run = gets_time() - t_bef_group
    print(f'Time to run group concatenation: {round(t_run/60, 3)} minutes!')

    return groups_df, track_df

def replacing_with_raw_data(groups_df, clean_name_to_uuid_dict, clean_name_to_raw_name_dict, df_thresholds):
    print('Replacing duplicated products output with raw data..')

    # mapping entity UUIDs
    groups_df['winner_entity_uuid'] = groups_df['winner'].map(clean_name_to_uuid_dict)
    groups_df['loser_entity_uuid'] = groups_df['loser'].map(clean_name_to_uuid_dict)

    # mapping raw item names
    groups_df['winner_name'] = groups_df['winner'].map(clean_name_to_raw_name_dict)
    groups_df['loser_name'] = groups_df['loser'].map(clean_name_to_raw_name_dict)

    # selecting columns
    output_df = groups_df.loc[:, ['winner_entity_uuid', 'winner_name', 'loser_entity_uuid', 'loser_name']]

    return output_df, groups_df

def duplicates_by_exact_product_name(df_nlp, groups_df, clean_name_to_raw_name_dict, output_df):
    print('Identifying entities sharing the same product name..')

    print(f"# cases: {df_nlp[df_nlp.duplicated(['raw_item_name'], keep=False)].shape[0]}")

    df_nlp[['item_uuid', 'raw_item_name', 'item_name', 'product_name']]

    # flagging winner products on duped set
    df_duped_name = df_nlp[df_nlp.duplicated(['raw_item_name'], keep=False)][['item_uuid', 'raw_item_name']].reset_index(drop=True)
    entities_already_winners = list(set(groups_df['winner_entity_uuid']))
    df_duped_name.loc[df_duped_name['item_uuid'].isin(entities_already_winners), 'is_winner'] = 1
    df_duped_name = df_duped_name.sort_values(['is_winner', 'raw_item_name']).reset_index(drop=True)

    # separating the first record of each set from the rest
    df_rest = df_duped_name[df_duped_name.duplicated(['raw_item_name'], keep='first')].sort_values('raw_item_name').reset_index(drop=True)
    first_duped_entities = list(set(df_rest['item_uuid']))
    df_first = df_duped_name[~df_duped_name['item_uuid'].isin(first_duped_entities)].sort_values('raw_item_name').reset_index(drop=True)

    # merging datasets to create winner to loser relationships
    df_first.columns = [f'first_{col_}' for col_ in df_first.columns]
    df_rest.columns = [f'rest_{col_}' for col_ in df_rest.columns]
    df_merge = df_first.merge(df_rest, how='inner', left_on='first_raw_item_name', right_on='rest_raw_item_name')

    # selecting columns and mapping to output df
    df_merge = df_merge.loc[:, ['first_item_uuid', 'first_raw_item_name', 'rest_item_uuid', 'rest_raw_item_name']]
    df_merge.rename(columns={'first_item_uuid': 'winner_entity_uuid', 'rest_item_uuid': 'loser_entity_uuid', 'first_raw_item_name': 'winner_name', 'rest_raw_item_name': 'loser_name'}, inplace=True)

    output_df = pd.concat([output_df, df_merge], axis=0).reset_index(drop=True)
    output_df = output_df.sort_values('winner_name').reset_index(drop=True)

    print(f'# duplicates / relations to handle: {output_df.shape[0]}')

    return output_df


def main():
    # Initial time
    t_initial = gets_time()

    # reading CSV file, cleaning parent_chain_name, and selecting data to work on
    data, item_name_image_dict = read_and_select()

    # NLP + regex product name cleaning --> new column: product_name
    df_nlp, clean_name_to_package_dict, clean_name_to_uuid_dict, clean_name_to_raw_name_dict = nlp_regex_cleaning(language_, data)

    # Appying TF-IDF method
    df_tf, tf_idf_matrix = tf_idf_method(df_nlp)

    # Applying cosine similarity to detect most similar products (potential group)
    matches_df = cosine_similarity_calculation(df_tf, tf_idf_matrix)

    # Calculating fuzzy ratios and keeping products with similarity above threshold_products
    df_similars = fuzzy_ratios(matches_df, clean_name_to_uuid_dict) 

    # extending product similarities: A similar to B, and B similar to D; then A, B, and D are similars
    df_similars_ext = extends_similarities(df_similars)

    # calculating fuzzy ratios between product packages, keeping similarities above threshold_package
    df_clean, df_thresholds = cleaning_by_package_similarity(df_similars_ext, clean_name_to_package_dict)

    # dictionaries to map product_name --> index
    product_index_dict, index_product_dict = creating_product_index_name_mapping_dict(df_tf)

    # product names into integers --> easy to compare
    df_clean = product_name_replacement(df_clean, product_index_dict)

    # concatenating groups to global dataframe
    groups_df, track_df = groups_concatenation(df_clean, index_product_dict)

    # replacing duplicated products output with raw data
    output_df, groups_df = replacing_with_raw_data(groups_df, clean_name_to_uuid_dict, clean_name_to_raw_name_dict, df_thresholds)
    
    # identifying entities sharing the same product name
    output_df = duplicates_by_exact_product_name(df_nlp, groups_df, clean_name_to_raw_name_dict, output_df)
    
    # saving results
    output_df.to_csv(f'~/Downloads/heuristic_duplicates/{threshold_products}_output_{country}_{parent_chain}_duplicates.csv', index=False)


if __name__ == "__main__":
    main()
