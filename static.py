import pandas as pd
import numpy as np
import re
import time
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import nltk.corpus

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

def gets_time():
    return time.time()

## Name cleaning: NLP + Regex

def clean_text(df, col_name, new_col_name):
    # column values to lower case
    df[new_col_name] = df[col_name].str.lower().str.strip()
    # removes special characters
    df[new_col_name] = df[new_col_name].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z.% \t])", "", x))
    return df

def replace_stop_words(df, col, stop_list):
    df['{}_stop'.format(col)] = df[col].apply(lambda x: ' '.join([word for word in x.split() if x not in stop_list]))
    return df

def word_lemmatizer(text):
    text_lemma = [WordNetLemmatizer().lemmatize(word) for word in text]
    return text_lemma

def nlp_cleaning(df, stop_words, regex_clean):
    # normalization
    df = clean_text(df, 'item_name', 'item_name_norm')
    # remove stop words
    df = replace_stop_words(df, 'item_name_norm', stop_words)
    # tokenize text
    df['item_name_token'] = df['item_name_norm_stop'].apply(lambda x: word_tokenize(x))
    # lemmatization
    df['item_name_token_lemma'] = df['item_name_token'].apply(lambda x: word_lemmatizer(x))
    # joining lemmas
    not_list = ['.']
    df['product_name'] = df['item_name_token_lemma'].apply(lambda list_: ' '.join([word for word in list_ if word not in not_list]))
    # cleaning product names with regex
    df['product_name'] = df['product_name'].apply(lambda x: re.sub(regex_clean, "", x))
    return df

def cosine_similarity(A, B, ntop, lower_bound=0):
    # force A and B as a compressed sparse row (CSR) matrix.
    # CSR --> efficient operations, fast matrix vector products
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)
    
    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'product_name': left_side,
                          'match': right_side,
                           'similarity_score': similairity})

def package_extract(df, column, regex_):
    """
    Extracts the package from a product name. Uses a regular expression for these.
    
    Inputs:
    - df: dataframe
    - column: product name column where to look for packages
    - regex_: regular expression formula to match patterns
    
    Output: a column with the package of the specified product name column
    """
    packs = df[column].str.extract(regex_)
    packs['package'] = packs[packs.columns[0:]].apply(lambda x: ','.join(x.dropna()), axis=1)
    packs = packs.loc[:, ['package']]
    return packs.loc[:, ['package']]

def create_group_track_df(groups_df, track_df, product, applicants_list):
    if groups_df.shape[0] == 0:
        group_id = 0
    else:
        group_id = groups_df['group_id'].max() + 1
    if track_df.shape[0] == 0:
        track_id = 0
    else:
        track_id = track_df['group_id'].max() + 1
        
    df_temp_group = pd.DataFrame({
        'group_id': group_id,
        'leader': product,
        'member': applicants_list
        })
    df_temp_track = pd.DataFrame({
        'group_id': track_id,
        'member': applicants_list
        })
    
    return df_temp_group, df_temp_track

def verify_and_concat_groups(groups_df, track_df, index_, applicants_list):
    # verify if any of the applicants is already assigned to a group, if not:    
    if track_df[track_df['member'].isin(applicants_list)].shape[0] == 0:
        # create df for the group
        tmp_group_df, tmp_track_df = create_group_track_df(groups_df, track_df, index_, applicants_list)
        # concat group to the global groups df
        groups_df = pd.concat([groups_df, tmp_group_df], axis=0).reset_index(drop=True)
        # concat track group to track global groups df
        track_df = pd.concat([track_df, tmp_track_df], axis=0).reset_index(drop=True)
    else:
        # get the group ids where all of the candidates are assigned
        groups_id_list = list(track_df[track_df['member'].isin(applicants_list)]['group_id'].unique())
        # locate where the group is
        select_df = groups_df[groups_df['group_id'].isin(groups_id_list)]
        # list of actual members of the group
        already_members = list(pd.unique(select_df[['leader', 'member']].values.ravel('K')))
        # union of already members + apliccants list --> idea: get a unique selection of a wider spectrum
        concatenated_list = list(set(already_members + applicants_list))
        # remove group from global groups and track dataframes
        groups_df = groups_df[~groups_df['group_id'].isin(groups_id_list)].copy()
        track_df = track_df[~track_df['group_id'].isin(groups_id_list)]
        # re-create both: groups & track - global dfs
        tmp_group_df, tmp_track_df = create_group_track_df(groups_df, track_df, index_, concatenated_list)
        # add the new set to both: groups & track - global dfs
        groups_df = pd.concat([groups_df, tmp_group_df], axis=0).reset_index(drop=True)
        track_df = pd.concat([track_df, tmp_track_df], axis=0).reset_index(drop=True)
    return groups_df, track_df

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
    
    # ### Create a match table to show the similarity scores
    matches_df = pd.DataFrame()
    matches_df = get_matches_df(matches, df_tf['product_name'], top=False)
    matches_df = matches_df.drop_duplicates().reset_index(drop=True)

    return matches_df

def fuzzy_ratios(matches_df, threshold_products):
    print('Fuzzy ratios calculation..')
    print(f'Product Threshold: {threshold_products}')
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
    df_similars_mrg = df_similars.merge(df_similars_copy, how='inner', on='match')
    df_similars_mrg.drop('similarity_score', axis=1, inplace=True)

    # melt dataframe
    df_melt = df_similars_mrg.melt(id_vars=['product_name', 'fuzz_ratio'], var_name='which_match', value_name='candidate')
    df_melt = df_melt.drop('which_match', axis=1)[['product_name', 'candidate', 'fuzz_ratio']]

    df_similars_ext = df_melt.drop_duplicates(['product_name', 'candidate']).sort_values(by=['product_name', 'candidate'])\
                .reset_index(drop=True)
    
    return df_similars_ext

def cleaning_by_package_similarity(df_similars_ext, threshold_package, match_col='candidate', return_barcode=False):
    print('Filtering product matches by package fuzzy ratio similarity measure..')
    reg_promos = r'(\d+x\d+\w+)|(\d+ x \d+\w+)|(\d+ x \d+ \w+)|(\d+\w+ x \d+ \w+)|(\d+ x \d+\.\d+\w+)|(\d+ x \d+\.\d+ \w+)|(x \d+)|(x \d+g)|(x \d+ g)|(x\d+)|(\d+\w+ \d+pk)|(\d+\w+ \d+pack)|(\d+\w+ \d+ pk)|(\d+\w+ \d+ pack)|(\d+ pack)|(\d+ pk)|(x\d+ \d+g)|(x\d+ \d+0g)|'
    reg_pack = r'(\d+\.+\d+\w+)|(\d+\.+\d+ \w+)|(\d+ ml)|(\d+ g)|(\d+\w+)|(\d+ \w+)|(0\.\d+ litre)|(\d+\.\d+ litre)|(0\.\d+l)|(\d+\.\d+ l)|(\d+\.\d+l)|(\d+l)|(\d+ cl)|(\d+cl)|(\d+0 cl)|(\d+\.\d+ kg)|(\d+ ml)|(\d+ kilo)|'
    reg_pieces = r'(\d+ piece)|(\d+0 piece)|(\d+piece)|(\d+ piezas)|'
    reg_sizes = r'(\d+ inch)|'
    reg_med = r'(\d+ mg)|'
    reg_in = r'(\d+ in \d+)'

    reg_package = reg_promos + reg_pack  + reg_pieces + reg_sizes + reg_med + reg_in
    # extracting package
    df_similars_ext['package'] = package_extract(df_similars_ext, 'product_name', reg_package)
    df_similars_ext['package_candidate'] = package_extract(df_similars_ext, match_col, reg_package)
    # package similarity
    df_similars_ext['package_ratio'] = df_similars_ext.apply(lambda x: fuzz.token_sort_ratio(x['package'],\
                                                                                x['package_candidate']), axis=1)
                                                                                
    # Package filter + Column selection
    print(f'Package Threshold: {threshold_package}')
    df_clean = df_similars_ext[df_similars_ext['package_ratio'] > threshold_package].reset_index(drop=True)

    if return_barcode:
        return df_clean.loc[:, ['product_name', match_col, 'similarity_score', 'fuzz_ratio', 'package_ratio']]
    else:
        return df_clean.loc[:, ['product_name', match_col]]

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
    print('Concatenating groups to global DF..')
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

def remove_duplication_for_uuid(data):
    print(f"UUIDs may be assigned to more than a single product; Fixing this issue..")
    
    # identifies the existance of uuids assigned to more than 1 item name
    identify_duplication_df = data.groupby('item_uuid').agg({'item_name': 'count'}).reset_index().sort_values(by='item_name', ascending=False).reset_index(drop=True)
    number_uuids_more_than_1 = identify_duplication_df[identify_duplication_df['item_name'] > 1].drop_duplicates('item_uuid').reset_index(drop=True).shape[0]
    print(f"Number of UUIDs assigned to more than 1 product: {number_uuids_more_than_1}")

    # aggregates and sorts values
    data['number_sku_sold'] = 1
    duplicated_df = data.groupby(['item_uuid', 'item_name']).agg({'number_sku_sold': sum}).reset_index()
    duplicated_df = duplicated_df.sort_values(by=['item_uuid', 'number_sku_sold'], ascending=False).reset_index(drop=True)

    # removes duplicated item names --> idea: keep the item name with the higher number of sales
    duplicated_df = duplicated_df.drop_duplicates('item_uuid').reset_index(drop=True)

    duplicated_unique_uuids_list = list(set(duplicated_df['item_uuid']))
    print(f'Missing UUIDs after removing duplicated assingments: {len(list(set(data[~data["item_uuid"].isin(duplicated_unique_uuids_list)]["item_uuid"])))}')
    
    print(f'Dataframe shape at this stage of the process (remove duplicated uuids): {duplicated_df.shape}')

    # print number of duplicated (same as initial of function)
    
    return duplicated_df

