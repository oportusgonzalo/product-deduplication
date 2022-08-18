
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

# parameters
country = ''
parent_chain = ''

# hiperparameters
threshold_products = 85
threshold_package = 75


def classification_accuracy(df_clean):
    print('Calculating the number of correct group assignments..')
    df_true = df_clean[~df_clean['label'].isna()]
    df_true.label = df_true.label.str.strip().str.lower()
    print(f'Number of correct assignments: {round(df_true[df_true.label == "true"].shape[0]/len(df_true), 3)}')

def map_member_to_item_uuid(df_links):
    print('Mapping backwards each member to the UUID..')
    # open back propagation file
    df_back_propagation = pd.read_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')
    df_back_propagation.drop('image_url', axis=1, inplace=True)

    # cleaning the set
    df_links = df_links.drop_duplicates('member').reset_index(drop=True)

    # merge back propagation with links to have: | item_uuid | item_name | canonical_id | canonical_leader | canonical_member |   
    new_links_df = df_links.merge(df_back_propagation, how='inner', left_on='member', right_on='product_name')
    new_links_df.rename(columns={'member': 'canonical_member'}, inplace=True)
    new_links_df = new_links_df.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member', 'agent_verified']]
    
    # stats
    new_links = new_links_df.canonical_member.unique()
    print(f'Number of links missing: {len(df_links[~df_links["member"].isin(new_links)]["member"].unique())}')
    print(f'Number of unique members at this stage of the process (mapping members to uuid): {df_links.drop_duplicates("member").shape[0]}')

    return new_links_df

def assign_ids_to_candidates(canonical_df, df_canonical_candidate):
    max_canonical_id = canonical_df['canonical_id'].max()
    canonical_leaders = canonical_df['canonical_leader'].unique()
    # canonical_leader - canonical_id dictionary
    leader_name_id_dict = dict(zip(canonical_df['canonical_leader'], canonical_df['canonical_id']))

    # assign random and unique ID
    df_canonical_candidate.insert(0, 'canonical_id', range(max_canonical_id, max_canonical_id + len(df_canonical_candidate)))
    # for leaders that have been already match in the canonical file, we map the IDs
    for canonical_name, id_ in leader_name_id_dict.items():
        df_canonical_candidate.loc[df_canonical_candidate['canonical_leader'] == canonical_name, 'canonical_id'] = id_

    # dictionary to assign IDs on link dataset
    candidate_name_id_dict = dict(zip(df_canonical_candidate['canonical_leader'], df_canonical_candidate['canonical_id']))
    
    return df_canonical_candidate, candidate_name_id_dict

def standardize_format(new_canonical_data, new_canonical_links):
    for col in ['canonical_leader', 'brand', 'name']:
        new_canonical_data[col] = new_canonical_data[col].str.title()
    
    for col in ['canonical_leader', 'canonical_member']:
        new_canonical_links[col] = new_canonical_links[col].str.title()
    
    return new_canonical_data, new_canonical_links

def new_leaders_to_canonical(df_canonical_candidate):
    pass

def main():

    df = pd.read_csv(f'agents_clean/agents_clean_{country}_{parent_chain}.csv')
    df.columns = df.columns.str.strip().str.lower()

    for col in ['leader', 'member', 'label', 'canonical_leader', 'brand', 'name', 'package']:
        df[col] = df[col].str.strip().str.lower()

    # Number of correct assignments
    classification_accuracy(df)

    # canonical database structure
    print(f'Agents output shape: {df.shape}')
    print(f'Initial number of members: {len(df["member"].unique())}')

    df_canonical_candidate = df.loc[:, ['canonical_leader', 'brand', 'name', 'package', 'promotion']]
    df_canonical_candidate = df_canonical_candidate.drop_duplicates('canonical_leader').reset_index(drop=True)
    df_canonical_candidate['agent_verified'] = 1
   
    print(f'Number of unique leaders: {len(df_canonical_candidate["canonical_leader"].unique())}')
    print(f'Percentage of unique products: {round(len(df_canonical_candidate["canonical_leader"].unique())/len(df["member"].unique()), 3)}')
    
    # links database structure
    df_links = df.loc[:, ['member', 'canonical_leader']]
    df_links['agent_verified'] = 1

    # ask for existance of a canonical catalog file
    if os.path.exists('canonical_data/canonical_catalog.csv'):
        print('It does exist a canonical catalog..')
        # adds new canonical data to global canonical dataframe
        print('Reading canonical file..')
        canonical_df = pd.read_csv('canonical_data/canonical_catalog.csv')

        # assign correct IDs to candidates
        df_canonical_candidate, candidate_name_id_dict = assign_ids_to_candidates(canonical_df, df_canonical_candidate)
        print('Concatenating new canonical products with actual canonical products..')
        new_canonical_df = pd.concat([canonical_df, df_canonical_candidate], axis=0).reset_index(drop=True)
        new_canonical_df = new_canonical_df.drop_duplicates().reset_index(drop=True)

        # adds new links to global links file
        print('Reading canonical links file..')
        canonical_links_df = pd.read_csv('canonical_data/canonical_links.csv')
        df_links['canonical_id'] = df_links['canonical_leader'].map(candidate_name_id_dict)

        # mapping members back to item_uuid: | item_uuid | item_name | canonical_id | canonical_leader | canonical_member |
        new_links_df = map_member_to_item_uuid(df_links)
        print('Concatenating new canonical links with actual canonical links..')
        new_canonical_links_df = pd.concat([canonical_links_df, new_links_df], axis=0).reset_index(drop=True)
        new_canonical_links_df = new_canonical_links_df.drop_duplicates().reset_index(drop=True)

        # better format to dataframes
        new_canonical_df, new_canonical_links_df = standardize_format(new_canonical_df, new_canonical_links_df)
        
        # saving datasets
        print('Saving updated canonical files..')
        new_canonical_df.to_csv('canonical_data/canonical_catalog.csv', index=False)
        new_canonical_links_df.to_csv('canonical_data/canonical_links.csv', index=False)

    else:
        print('There is no canonical catalog..')
        # create canonical ID from scratch --> structure
        df_canonical_candidate.insert(0, 'canonical_id', range(0, len(df_canonical_candidate)))

        # dictionary to assign IDs on link dataset
        candidate_name_id_dict = dict(zip(df_canonical_candidate['canonical_leader'], df_canonical_candidate['canonical_id']))
        df_links['canonical_id'] = df_links['canonical_leader'].map(candidate_name_id_dict)
        df_links = df_links.loc[:, ['member', 'canonical_id', 'canonical_leader', 'agent_verified']]

        # mapping members back to item_uuid: | item_uuid | item_name | canonical_id | canonical_leader | canonical_member |
        new_links_df = map_member_to_item_uuid(df_links)

        # better format to dataframes
        df_canonical_candidate, new_links_df = standardize_format(df_canonical_candidate, new_links_df)

        # saving datasets
        print('Saving canonical files..')
        df_canonical_candidate.to_csv('canonical_data/canonical_catalog.csv', index=False)
        new_links_df.to_csv('canonical_data/canonical_links.csv', index=False)
    
    print('Success!')


if __name__ == "__main__":
    main()