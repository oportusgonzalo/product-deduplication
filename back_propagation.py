import pandas as pd

# define parameters
country = 'uk'
parent_chain = 'nisa'
threshold_products = 85
threshold_package = 75
exists_canonical_catalog = True

# canonical file
canonical_file = 'canonical_catalog'

def classification_accuracy(df_clean):
    print('Calculating the number of correct group assignments..')
    df_true = df_clean[~df_clean['label'].isna()]
    df_true.label = df_true.label.str.strip().str.lower()
    print(f'Number of correct assignments: {round(df_true[df_true.label == "true"].shape[0]/len(df_true), 3)}')

def mapping_agents_cleaned_items_to_raw():
    # reading file that has item_name and product_name (post NLP + regex cleaning)
    df_back = pd.read_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')

    # agents cleaned groups
    df_clean = pd.read_csv(f'agents_clean/agents_clean_{country}_booker.csv') # MODIFY WHEN POSSIBLE
    df_clean.columns = df_clean.columns.str.strip().str.lower()

    # Number of correct assignments
    classification_accuracy(df_clean)

    # useful columns
    df_clean = df_clean.loc[: ,['member', 'canonical_leader', 'brand', 'name', 'package', 'promotion']]

    # merging datasets
    df_merge = df_back.merge(df_clean, how='inner', left_on='product_name', right_on='member')
    df_merge.drop(['product_name', 'member'], axis=1, inplace=True)
    
    return df_merge

def direct_links():
    print('Reading file with products that have a direct link to the canonical catalog..')
    # reading file with direct matches
    df_direct = pd.read_csv(f'bivariate_outputs/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv')
    df_direct = df_direct.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader']]
    return df_direct

def add_leaders_to_canonical(df_merge):
    # reading canonical file
    canonical_data = pd.read_csv(f'canonical_data/{canonical_file}.csv')
    canonical_data.columns = canonical_data.columns.str.strip().str.lower()
    canonical_data.rename(columns={'group_id': 'canonical_id'}, inplace=True)
    canonical_data = canonical_data.loc[:, ['canonical_id', 'canonical_leader', 'brand', 'name', 'package', 'promotion']]

    # format to new products dataframe
    df_merge_new = df_merge.loc[:, ['canonical_leader', 'brand', 'name', 'package', 'promotion']]
    df_merge_new['canonical_id'] = range(int(canonical_data['canonical_id'].max()), int(canonical_data['canonical_id'].max()) + len(df_merge))

    # new canonical set
    df_new_canonical = pd.concat([canonical_data, df_merge_new], axis=0).reset_index(drop=True)
    df_new_canonical = df_new_canonical.drop_duplicates().reset_index(drop=True)
    
    return df_new_canonical


def main():
    # mapping products cleaned by agents to raw item names / IDs
    df_merge = mapping_agents_cleaned_items_to_raw()

    if exists_canonical_catalog:
        # reading direct links file
        df_direct = direct_links()
        # extract new links
        df_new_links = df_merge.loc[:, ['item_uuid', 'item_name', 'canonical_leader']]
        df_new_links['canonical_id'] = range(df_direct['canonical_id'].max(), df_direct['canonical_id'].max() + len(df_new_links))
        # concatenate links
        df_links = pd.concat([df_direct, df_new_links], axis=0).reset_index(drop=True)
        df_links = df_links.drop_duplicates().reset_index(drop=True)

        # adding new leaders to canonical database
        df_new_canonical = add_leaders_to_canonical(df_merge)

    else:
        # creates dataframe with links
        df_links = df_merge.loc[:, ['item_uuid', 'item_name', 'canonical_leader']]
        df_links['canonical_id'] = range(0, + len(df_links))
        df_links = df_links.drop_duplicates().reset_index(drop=True)

        # adding new leaders to canonical database
        df_new_canonical = add_leaders_to_canonical(df_merge)


    print(df_links)
    print(df_new_canonical)

    '''
    NOTE: need to deliver two files.

    1) Map between raw item_name and canonical_id/canonical_leader: |item_uuid|item_name|canonical_id|canonical_leader| (READY)
    2) Canonical catalog with all products (old + new): |canonical_id|canonical_leader|brand|name|package|promotion|

    * Give better format to leaders: title(), etc.
    '''

    
if __name__ == "__main__":
    main()