
import pandas as pd


country = ''


def stats_(df_links, time_):
    # stats to validate
    print(f'N° {time_} canonical IDs: {len(list(set(df_links["canonical_id"])))}')
    print(f'N° {time_} canonical leaders: {len(list(set(df_links["canonical_id"])))}')
    print(f'Initial shape: {df_links.shape}')

def fix_nan(df_links, df_na):
    print('Fixing columns with nan values on canonical leader column..')

    df_na['canonical_leader'] = df_na['canonical_member']
    max_id = int(df_links['canonical_id'].max())
    na_leaders_list = list(set(df_na['canonical_leader']))
    na_leaders_id_dict = dict(zip(na_leaders_list, range(max_id + 1, max_id + len(na_leaders_list) + 1)))
    df_na['canonical_id'] = df_na['canonical_leader'].map(na_leaders_id_dict)
    
    return df_na

def removing_duplicated_assignments(df_links, df_dup):
    print('Fixing canonical leaders assigned to more than one canonical ID..')
    df_dup = df_dup.drop_duplicates('canonical_leader').reset_index(drop=True)
    dup_dict = dict(zip(df_dup['canonical_leader'], df_dup['canonical_id']))

    for item_, canon_id in dup_dict.items():
        df_links.loc[df_links['canonical_leader'] == item_, 'canonical_id'] = canon_id

    return df_links


def main():

    df_links = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')

    # stats to validate
    stats_(df_links, 'initial')

    # splitting the dataframe: set with nan, set with others
    df_na = df_links[df_links['canonical_leader'].isna()].reset_index(drop=True)
    df_links = df_links[~df_links['canonical_leader'].isna()].reset_index(drop=True)

    df_links['canonical_leader'] = df_links['canonical_leader'].str.strip().str.lower()
    df_dup = df_links.loc[:, ['canonical_id', 'canonical_leader']].drop_duplicates().reset_index(drop=True)

    print(f'N° leaders with duplicated ID: {df_dup[df_dup["canonical_leader"].duplicated() == True].shape[0]}')

    # verifying which canonical leaders are assigned to more than one canonical ID
    df_dup = df_dup[df_dup.duplicated(['canonical_leader'], keep=False)].sort_values('canonical_leader').reset_index(drop=True)

    if df_dup.shape[0] > 0:
        df_links = removing_duplicated_assignments(df_links, df_dup)
    
    # fixing null values in canonical_id and canonical_leader
    df_na = fix_nan(df_links, df_na)

    # concat to links
    df_links = pd.concat([df_links, df_na], axis=0).reset_index(drop=True)

    # stats to validate
    stats_(df_links, 'final')

    # saving result
    df_links.to_csv(f'canonical_data/{country}/{country}_canonical_links.csv', index=False)


if __name__ == "__main__":
    main()