import pandas as pd
import numpy as np
import re
import time

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
    df['product_name'] = df['item_name_token_lemma'].apply(lambda list_: ' '.join([word for word in list_]))
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

