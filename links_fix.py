import pandas as pd

country = ''

def main():

    df_links = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')
    df_na = df_links[df_links['canonical_leader'].isna()].drop_duplicates('canonical_member').reset_index(drop=True)
    df_links = df_links[~df_links['canonical_leader'].isna()].reset_index(drop=True)
    df_links['canonical_leader'] = df_links['canonical_leader'].str.strip()

    df_dup = df_links.loc[:, ['canonical_id', 'canonical_leader']].drop_duplicates().reset_index(drop=True)
    df_dup = df_dup[df_dup.duplicated(['canonical_leader'], keep=False)].sort_values('canonical_leader').reset_index(drop=True)
    df_dup = df_dup.drop_duplicates('canonical_leader').reset_index(drop=True)
    dup_dict = dict(zip(df_dup['canonical_leader'], df_dup['canonical_id']))

    for item_, canon_id in dup_dict.items():
        df_links.loc[df_links['canonical_leader'] == item_, 'canonical_id'] = canon_id
    
    # fixing null values in canonical_id and canonical_leader
    df_na['canonical_leader'] = df_na['canonical_member']
    max_id = int(df_links['canonical_id'].max())
    df_na['canonical_id'] = range(max_id + 1, max_id + df_na.shape[0] + 1)
    
    # concat to links
    df_links = pd.concat([df_links, df_na], axis=0).reset_index(drop=True)

    # stats to validate
    print(f'N° canonical IDs: {len(list(set(df_links["canonical_id"])))}')
    print(f'N° canonical leaders: {len(list(set(df_links["canonical_id"])))}')

    # saving result
    df_links.to_csv(f'canonical_data/{country}/{country}_canonical_links.csv', index=False)


if __name__ == "__main__":
    main()