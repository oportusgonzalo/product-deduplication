
import pandas as pd
import numpy as np
import os

from static import *

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords


country = ''
parent_chain = ''
language_ = ''


def reading_files():
    print('Reading files..')

    df_canonical = pd.read_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv')
    df_links = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')
    df_missing = pd.read_csv(f'missing/{country}/{country}_missing_{parent_chain}.csv')

    return df_canonical, df_links, df_missing


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


def processing_direct_matches_to_canonical(df_links, data_nlp):
    print('Removing missing members already linked to canonical..')

    # basic cleaning for consistency
    df_links['canonical_member'] = df_links['canonical_member'].str.strip().str.lower()
    data_nlp['product_name'] = data_nlp['product_name'].str.strip().str.lower()

    # removing UUIDs already linked to canonical
    canonical_uuids_list = list(set(df_links['item_uuid']))
    df_clean = data_nlp[~data_nlp['item_uuid'].isin(canonical_uuids_list)].reset_index(drop=True)
    
    # identifying members already linked to canonical products (Idea: assign canonical_ids so we don't introduce duplicated items)
    canonical_members_list = list(set(df_links['canonical_member']))
    df_members_on_canonical = df_clean[df_clean['product_name'].isin(canonical_members_list)].reset_index(drop=True)
    df_members_not_canonical = df_clean[~df_clean['product_name'].isin(canonical_members_list)].reset_index(drop=True)
    print(f'Verification - members on canonical + members not on canonical vs total applicant members: {(df_members_on_canonical.shape[0] + df_members_not_canonical.shape[0])}  vs {df_clean.shape[0]}')

    return df_links, df_members_on_canonical, df_members_not_canonical

    
def canonical_catalog_concatenation(df_canonical, df_members_not_canonical):
    print('Adding missing members to the Canonical Catalog..')

    # selecting clean product name column --> will be appended to leaders column
    df_members = df_members_not_canonical.loc[:, ['product_name']].rename(columns={'product_name': 'canonical_leader'})

    # adding canonical columns
    for col_ in ['brand', 'name', 'package', 'promotion']:
        df_members[col_] = np.nan
    df_members['agent_verified'] = 0

    # adding canonical IDs to members not linked (before concatenating) --> ID starts from max(canonical_id) + 1
    max_canonical_id = int(df_canonical['canonical_id'].max())
    df_members.insert(0, 'canonical_id', range(max_canonical_id + 1, max_canonical_id + len(df_members) + 1))

    # concatenating canonical dataframe with new members
    new_canonical_df = pd.concat([df_canonical, df_members], axis=0).reset_index(drop=True)
    
    return new_canonical_df, df_members


def linking_new_members_to_canonical(new_canonical_df, df_links, df_members_on_canonical, df_members_not_canonical):
    print('Adding new members to canonical links file..')

    # dictionaries to map members
    canonical_leader_id_dict = dict(zip(new_canonical_df['canonical_leader'], new_canonical_df['canonical_id']))
    canonical_verified_id_dict = dict(zip(new_canonical_df['canonical_id'], new_canonical_df['agent_verified']))
    links_member_id_dict = dict(zip(df_links['canonical_member'], df_links['canonical_id']))

    # members on canonical have canonical ids already assigned --> mapping from links file
    df_members_on_canonical['canonical_id'] = df_members_on_canonical['product_name'].map(links_member_id_dict)
    df_members_on_canonical.rename(columns={'product_name': 'canonical_member'}, inplace=True)
    print(f'N° members on canonical with NaNs on canonical ID: {df_members_on_canonical[df_members_on_canonical["canonical_id"].isna()].shape[0]}')

    # members not on canonical have been assigned canonical ids when concatenating to canonical --> assigning from canonical file
    df_members_not_canonical['canonical_id'] = df_members_not_canonical['product_name'].map(canonical_leader_id_dict)
    df_members_not_canonical.rename(columns={'product_name': 'canonical_member'}, inplace=True)
    print(f'N° members not on canonical with NaNs on canonical ID: {df_members_not_canonical[df_members_not_canonical["canonical_id"].isna()].shape[0]}')

    # concatenating all new members to be added to links file
    df_applicant_links = pd.concat([df_members_on_canonical, df_members_not_canonical], axis=0).reset_index(drop=True)
    
    # adding canonical leader column
    canonical_id_leader_dict = dict(zip(new_canonical_df['canonical_id'], new_canonical_df['canonical_leader']))
    df_applicant_links['canonical_leader'] = df_applicant_links['canonical_id'].map(canonical_id_leader_dict)

    # adding agent verified column
    df_applicant_links['agent_verified'] = df_applicant_links['canonical_id'].map(canonical_verified_id_dict)

    # re-organizing set
    df_applicant_links = df_applicant_links.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member', 'agent_verified']]

    # concatenating to global links set
    df_new_links = pd.concat([df_links, df_applicant_links], axis=0).reset_index(drop=True)

    return df_new_links


def main():

    # reading files to be transformed
    df_canonical, df_links, df_missing = reading_files()

    # applying NLP to get direct matches on canonical and reduce duplication when concatenating
    data_nlp = nlp_regex_cleaning(language_, df_missing)

    # removes direct matches to canonical by canonical member on links file
    df_links, df_members_on_canonical, df_members_not_canonical = processing_direct_matches_to_canonical(df_links, data_nlp)

    # adding missing members to the canonical catalog
    new_canonical_df, df_members = canonical_catalog_concatenation(df_canonical, df_members_not_canonical)
    
    # adding new members to canonical links file
    df_new_links = linking_new_members_to_canonical(new_canonical_df, df_links, df_members_on_canonical, df_members_not_canonical)

    # saving outputs
    if not os.path.isdir(f'missing/{country}'):
        os.mkdir(f'missing/{country}')

    new_canonical_df.to_csv(f'missing/{country}/{country}_canonical_catalog.csv', index=False)
    df_new_links.to_csv(f'missing/{country}/{country}_canonical_links.csv', index=False)
    


if __name__ == "__main__":
    main()
