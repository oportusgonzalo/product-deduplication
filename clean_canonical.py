
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

# parameters
country = 'ads_usa'
parent_chain = 'nestle'

# hiperparameters
threshold_products = 85
threshold_package = 75


def classification_accuracy(df_clean):
    print('Calculating the number of correct group assignments..')
    df_true = df_clean[~df_clean['label'].isna()]
    df_true.label = df_true.label.str.strip().str.lower()
    print(f'Number of correct assignments: {round(df_true[df_true.label == "true"].shape[0]/len(df_true), 3)}')

# maps to uuid in the case there's no canonical catalog
def map_member_to_item_uuid(df_links):
    print('Mapping backwards each member to the UUID..')
    # open back propagation file
    df_back_propagation = pd.read_csv(f'back_propagation/{country}/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')
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
    max_canonical_id = int(canonical_df['canonical_id'].max())

    # assign unique IDs
    df_canonical_candidate.insert(0, 'canonical_id', range(max_canonical_id + 1, max_canonical_id + len(df_canonical_candidate) + 1))
    
    # dictionary to assign IDs on link dataset
    candidate_name_id_dict = dict(zip(df_canonical_candidate['canonical_leader'], df_canonical_candidate['canonical_id']))
    
    return df_canonical_candidate, candidate_name_id_dict

def standardize_format(new_canonical_data, new_canonical_links):
    print(f'Standardizing strings on each dataframe..')
    for col in ['canonical_leader', 'brand', 'name']:
        new_canonical_data[col] = new_canonical_data[col].str.title()
    
    for col in ['canonical_leader', 'canonical_member']:
        new_canonical_links[col] = new_canonical_links[col].str.title()
    
    return new_canonical_data, new_canonical_links

def initial_cleaning_to_standardize(df):
    print('Initial Cleaning..')

    # splitting sets to clean nans
    df_na = df[df['canonical_leader'].isna()]
    print(f'N° rows with NaN: {df_na.shape[0]}')
    df = df[~df['canonical_leader'].isna()]

    # leaders will be equal to members
    df_na['canonical_leader'] = df_na['member']

    # concat to global set
    df = pd.concat([df, df_na], axis=0).reset_index(drop=True)

    return df


def nan_members_to_name(df_fixing):
    print('Replacing members with nan value but true assignment by leader..')
    df_fixing.loc[(df_fixing['member'].isna())&(df_fixing['label'] == 'true'), 'member'] = df_fixing.loc[(df_fixing['member'].isna())&(df_fixing['label'] == 'true'), 'canonical_leader']
    return df_fixing

def cleaning_non_pareto_dataframe(df_non_pareto):
    print('Cleaning the non-pareto set..')
    std_non_pareto_df = df_non_pareto.loc[:, ['leader']].drop_duplicates().reset_index(drop=True)
    std_non_pareto_df.rename(columns={'leader': 'canonical_leader'}, inplace=True)

    for col in ['brand', 'name', 'package']:
        std_non_pareto_df[col] = np.nan

    std_non_pareto_df['promotion'] = float(0)
    std_non_pareto_df['agent_verified'] = 0

    return std_non_pareto_df

def canonical_catalog_concatenation(df_canonical_candidate, df_non_pareto, canonical_df):
    print('Adding new leaders to canonical catalog..')

    # standardizing non pareto dataframe
    std_non_pareto_df = cleaning_non_pareto_dataframe(df_non_pareto)

    # concatenating sets
    df_potential_canonical = pd.concat([df_canonical_candidate, std_non_pareto_df], axis=0).reset_index(drop=True)
    df_potential_canonical['canonical_leader'] = df_potential_canonical['canonical_leader'].str.strip().str.lower()
    df_potential_canonical = df_potential_canonical.sort_values(['canonical_leader', 'brand', 'name', 'package']).reset_index(drop=True)
    df_potential_canonical = df_potential_canonical.drop_duplicates('canonical_leader').reset_index(drop=True)
    
    # we verify is any candidate leader is already on the canonical set and remove it from the candidates
    potential_leaders = list(set(df_potential_canonical['canonical_leader']))
    canonical_df['canonical_leader'] = canonical_df['canonical_leader'].str.strip().str.lower() 
    potential_already_leaders = canonical_df[canonical_df['canonical_leader'].isin(potential_leaders)]['canonical_leader'].drop_duplicates().reset_index(drop=True)

    if len(potential_already_leaders) > 0:
        print(f'Number of existant canonical candidates in canonical set: {len(potential_already_leaders)}')
        df_potential_canonical = df_potential_canonical[~df_potential_canonical['canonical_leader'].isin(potential_already_leaders)].reset_index(drop=True)
    else:
        print(f'There are no canonical candidates in the canonical set..')

    # structure candidates dataframe to be concatenated with canonical set
    if df_potential_canonical.shape[0] > 0:
        df_potential_canonical, candidate_name_id_dict = assign_ids_to_candidates(canonical_df, df_potential_canonical)
        print('Concatenating new canonical products with actual canonical products..')
        new_canonical_df = pd.concat([canonical_df, df_potential_canonical], axis=0).reset_index(drop=True)
        return new_canonical_df, candidate_name_id_dict
    else:
        candidate_name_id_dict = 0
        new_canonical_df = canonical_df.copy()
        return new_canonical_df, candidate_name_id_dict

def remove_direct_matches_on_canonical_links(canonical_links_df, df_direct):
    print('Removing direct matches UUIDs already on canonical links..') 
    
    canonical_links_uuids_list = list(set(canonical_links_df['item_uuid']))
    print(f'N° direct matches UUIDs already on canonical links: {df_direct[df_direct["item_uuid"].isin(canonical_links_uuids_list)].shape[0]}')
    df_direct = df_direct[~df_direct['item_uuid'].isin(canonical_links_uuids_list)].reset_index(drop=True)
    
    return canonical_links_df, df_direct

def ensuring_leaders_id_on_direct_set(canonical_links_df, df_to_fix):
    print(f'Making sure assigned canonical IDs are the same as in canonical links file..')

    leaders_id_dict = dict(zip(canonical_links_df['canonical_leader'], canonical_links_df['canonical_id']))
    
    for leader_, id_ in leaders_id_dict.items():
        df_to_fix.loc[df_to_fix['canonical_leader'] == leader_, 'canonical_id'] = id_
    
    df_to_fix = df_to_fix.drop_duplicates().reset_index(drop=True)

    return df_to_fix


def links_concatenation(canonical_links_df, df_direct, df_back, df_non_pareto, df_links, updated_canonical_dict):
    print('Adding new links to canonical links table..')

    # concatenating direct DF to canonical links (only if is not empty)
    if df_direct.shape[0] > 0:
        df_direct['agent_verified'] = 1
        for col in ['canonical_leader', 'canonical_member']:
            df_direct[col] = df_direct[col].str.strip().str.lower()

        # removing direct matches UUIDs already on canonical links
        canonical_links_df, df_direct = remove_direct_matches_on_canonical_links(canonical_links_df, df_direct)
        # making sure direct matches assigned canonical IDs are the same as in canonical links file
        df_direct = ensuring_leaders_id_on_direct_set(canonical_links_df, df_direct)
        # concatenating sets
        new_canonical_links_df = pd.concat([canonical_links_df, df_direct], axis=0).reset_index(drop=True)
    else:
        new_canonical_links_df = canonical_links_df.copy()

    # selecting useful columns (more clarity of the join)
    df_back = df_back.drop('image_url', axis=1)
    df_non_pareto = df_non_pareto.drop(['group_id', 'modify_leader', 'image_url'], axis=1)
    df_non_pareto['agent_verified'] = 0

    # merge to match item_uuids with canonical ids (format: item_uuid-item_name-canonical_id-canonical-leader-canonical_member-agent-verified)
    df_non_pareto = df_non_pareto.merge(df_back, how='inner', left_on='member', right_on='product_name')
    df_links = df_links.merge(df_back, how='inner', left_on='member', right_on='product_name')
    
    # clean-up
    df_non_pareto.rename(columns={'leader': 'canonical_leader', 'member': 'canonical_member'}, inplace=True)
    df_links.rename(columns={'member': 'canonical_member'}, inplace=True)

    # making sure columns to map are standardized
    df_non_pareto['canonical_leader'] = df_non_pareto['canonical_leader'].str.strip().str.lower()
    df_links['canonical_leader'] = df_links['canonical_leader'].str.strip().str.lower()

    # using canonical IDs on the updated version of the canonical catalog to map on the links
    df_applicants = pd.concat([df_non_pareto, df_links], axis=0).reset_index(drop=True)
    df_applicants = df_applicants.drop_duplicates().reset_index(drop=True)
    df_applicants['canonical_id'] = df_applicants['canonical_leader'].map(updated_canonical_dict)

    # re-organizing
    df_applicants = df_applicants.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member', 'agent_verified']]

    # making sure applicants assigned canonical IDs are the same as in canonical links file
    df_applicants = ensuring_leaders_id_on_direct_set(new_canonical_links_df, df_applicants)

    # merging all into same dataframe
    new_canonical_links_df = pd.concat([new_canonical_links_df, df_applicants], axis=0).reset_index(drop=True)

    new_canonical_links_df = new_canonical_links_df.drop_duplicates('item_uuid').reset_index(drop=True)

    return new_canonical_links_df

def fixes_nan_on_canonical_links(new_canonical_links_df):
    print('Fixing NaN values on canonical ID / canonical leader columns..')

    # splitting data: set with nan's / others
    df_na = new_canonical_links_df[(new_canonical_links_df['canonical_id'].isna())|(new_canonical_links_df['canonical_leader'].isna())].reset_index(drop=True)
    print(f'N° of NaN IDs: {df_na["canonical_id"].drop_duplicates().shape[0]}')
    print(f'N° of NaN leaders: {df_na["canonical_leader"].drop_duplicates().shape[0]}')
    new_canonical_links_df = new_canonical_links_df[~(new_canonical_links_df['canonical_id'].isna())|~(new_canonical_links_df['canonical_leader'].isna())].reset_index(drop=True)

    # leaders will be the members
    df_na['canonical_leader'] = df_na['canonical_member']

    # replace on ID if the leader has been already mapped
    actual_ids_dict = dict(zip(new_canonical_links_df['canonical_leader'], new_canonical_links_df['canonical_id']))
    df_na['canonical_id'] = df_na['canonical_leader'].map(actual_ids_dict)
    
    # create ID for leaders that are still nan
    na_leaders_list = list(set(df_na[df_na['canonical_id'].isna()]['canonical_leader']))
    max_id = int(new_canonical_links_df['canonical_id'].max())
    na_leaders_id_dict = dict(zip(na_leaders_list, range(max_id + 1, max_id + len(na_leaders_list) + 1)))
    df_na['canonical_id'] = df_na['canonical_leader'].replace(na_leaders_id_dict)

    # verify if any of the nan leaders is already linked
    leaders_already_linked_list = list(set(new_canonical_links_df['canonical_leader']))
    leaders_to_fix = list(set(df_na.loc[df_na['canonical_leader'].isin(leaders_already_linked_list)]['canonical_leader']))
    for leader_ in leaders_to_fix:
        df_na.loc[df_na['canonical_leader'] == leader_, 'canonical_id'] = new_canonical_links_df.loc[new_canonical_links_df['canonical_leader'] == leader_, 'canonical_id']

    # concatenate both sets
    new_canonical_links_df = pd.concat([new_canonical_links_df, df_na], axis=0).reset_index(drop=True)

    return new_canonical_links_df

def add_missing_links_to_canonical(df_canonical_candidate, new_links_df):
    print('Adding new canonical items generated after fixing NaNs..')

    on_canonical_list = list(set(df_canonical_candidate['canonical_leader']))
    links_not_canonical_df = new_links_df[~new_links_df['canonical_leader'].isin(on_canonical_list)]

    if links_not_canonical_df.shape[0] > 0:
        df_to_add = new_links_df.loc[:, ['canonical_id', 'canonical_leader', 'agent_verified']]
        df_to_add[['brand', 'name', 'package', 'promotion']] = [np.nan, np.nan, np.nan, 0]
        df_to_add = df_to_add.loc[:, ['canonical_id', 'canonical_leader', 'brand', 'name', 'package', 'promotion', 'agent_verified']]
        df_canonical_candidate = pd.concat([df_canonical_candidate, df_to_add], axis=0).reset_index(drop=True)

    return df_canonical_candidate, new_links_df


def main():

    print(f'Processing {parent_chain.title()} merchant from {country.title()}..')

    df = pd.read_csv(f'agents_clean/{country}/agents_clean_{country}_{parent_chain}.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['label'] = df['label'].astype(str)

    for col in ['leader', 'member', 'label', 'canonical_leader', 'brand', 'name', 'package']:
        df[col] = df[col].str.strip().str.lower()

    # Number of correct assignments
    classification_accuracy(df)

    # canonical database structure
    print(f'Agents output shape: {df.shape}')
    print(f'Initial number of members: {len(df["member"].unique())}')

    # as we may have inconsistencies and nans we will pre-process the file
    df = initial_cleaning_to_standardize(df)

    # fixing members that have nan (after agents work)
    df = nan_members_to_name(df)

    df_canonical_candidate = df.loc[:, ['canonical_leader', 'brand', 'name', 'package', 'promotion']]
    df_canonical_candidate = df_canonical_candidate.drop_duplicates('canonical_leader').reset_index(drop=True)
    df_canonical_candidate['agent_verified'] = 1
   
    print(f'Number of unique leaders: {len(df_canonical_candidate["canonical_leader"].unique())}')
    print(f'Percentage of unique products: {round(len(df_canonical_candidate["canonical_leader"].unique())/len(df["member"].unique()), 3)}')
    
    # links database structure
    df_links = df.loc[:, ['member', 'canonical_leader', 'label']]
    df_links['agent_verified'] = 1
    df_links.drop('label', axis=1, inplace=True)

    # ask for existance of a canonical catalog file
    if os.path.exists(f'canonical_data/{country}/{country}_canonical_catalog.csv'):
        print('It does exist a canonical catalog..')

        # appending data to the canonical catalog
        print('Reading canonical file..')
        canonical_df = pd.read_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv')
        df_non_pareto = pd.read_csv(f'bivariate_outputs/{country}/{parent_chain}/bivariate_non_pareto_groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv')
        new_canonical_df, candidate_name_id_dict = canonical_catalog_concatenation(df_canonical_candidate, df_non_pareto, canonical_df)

        # dict to map leaders while building on canonical links
        updated_canonical_dict = dict(zip(new_canonical_df['canonical_leader'], new_canonical_df['canonical_id']))

        # appending data to the item canonical links DB; uses: canonical_links_df, df_direct, df_back, df_non_pareto, and df_links
        print('Reading canonical links file..')
        canonical_links_df = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')
        # standardize canonical links (if not generates duplicates)
        for col in ['canonical_leader', 'canonical_member']:
            canonical_links_df[col] = canonical_links_df[col].str.strip().str.lower()

        # when there are no direct matches, the file isn't created
        if os.path.exists(f'bivariate_outputs/{country}/{parent_chain}/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv'):
            df_direct = pd.read_csv(f'bivariate_outputs/{country}/{parent_chain}/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv')
        else:
            df_direct = pd.DataFrame()
        df_back = pd.read_csv(f'back_propagation/{country}/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')
        
        new_canonical_links_df = links_concatenation(canonical_links_df, df_direct, df_back, df_non_pareto, df_links, updated_canonical_dict)

        # better format to dataframes
        new_canonical_df, new_canonical_links_df = standardize_format(new_canonical_df, new_canonical_links_df)

        # fixing nan values on canonical ID or canonical leader
        new_canonical_links_df = fixes_nan_on_canonical_links(new_canonical_links_df)

        # as fixing nan's on links may lead to "new" canonical items, we must verify / add to canonical
        new_canonical_df, new_canonical_links_df = add_missing_links_to_canonical(new_canonical_df, new_canonical_links_df)

        # saving datasets
        print('Saving canonical files..')
        new_canonical_df.to_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv', index=False)
        new_canonical_links_df.to_csv(f'canonical_data/{country}/{country}_canonical_links.csv', index=False)
    
    else:
        print('There is no canonical catalog..')
        # create the country directory to save the file
        if not os.path.isdir(f'canonical_data/{country}'):
            os.mkdir(f'canonical_data/{country}')
        
        # adding non pareto dataframe to first version of canonical catalog
        if country in ['us&ca', 'ads_usa', 'ads_ca', 'ads_uk', 'ads_fr', 'ads_aus']:
            df_non_pareto = pd.read_csv(f'bivariate_outputs/{country}/{parent_chain}/bivariate_non_pareto_groups_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv')
             # standardizing non pareto dataframe
            std_non_pareto_df = cleaning_non_pareto_dataframe(df_non_pareto)

            # concatenating sets
            df_canonical_candidate = pd.concat([df_canonical_candidate, std_non_pareto_df], axis=0).reset_index(drop=True)
            df_canonical_candidate['canonical_leader'] = df_canonical_candidate['canonical_leader'].str.strip().str.lower()
            df_canonical_candidate = df_canonical_candidate.sort_values(['canonical_leader', 'brand', 'name', 'package']).reset_index(drop=True)
            df_canonical_candidate = df_canonical_candidate.drop_duplicates('canonical_leader').reset_index(drop=True)

        # create canonical ID from scratch --> structure
        df_canonical_candidate.insert(0, 'canonical_id', range(0, len(df_canonical_candidate)))

        # dictionary to assign IDs on link dataset
        candidate_name_id_dict = dict(zip(df_canonical_candidate['canonical_leader'], df_canonical_candidate['canonical_id']))

        # adding non pareto dataframe to first version of canonical links
        if country in ['us&ca', 'ads_usa', 'ads_ca', 'ads_uk', 'ads_fr', 'ads_aus']:
            # reading back propagation file
            df_back = pd.read_csv(f'back_propagation/{country}/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')

            # selecting useful columns (more clarity of the join)
            df_back = df_back.drop('image_url', axis=1)
            df_non_pareto = df_non_pareto.drop(['group_id', 'image_url'], axis=1)
            df_non_pareto['agent_verified'] = 0

            # merge to match item_uuids with canonical ids (format: item_uuid-item_name-canonical_id-canonical-leader-canonical_member-agent-verified)
            df_non_pareto = df_non_pareto.merge(df_back, how='inner', left_on='member', right_on='product_name')
            df_links = df_links.drop_duplicates('member').reset_index(drop=True)
            df_links = df_links.merge(df_back, how='inner', left_on='member', right_on='product_name')

            # clean-up
            df_non_pareto.rename(columns={'leader': 'canonical_leader', 'member': 'canonical_member'}, inplace=True)
            df_links.rename(columns={'member': 'canonical_member'}, inplace=True)

            # making sure columns to map are standardized
            df_non_pareto['canonical_leader'] = df_non_pareto['canonical_leader'].str.strip().str.lower()
            df_links['canonical_leader'] = df_links['canonical_leader'].str.strip().str.lower()

            # using canonical IDs on the updated version of the canonical catalog to map on the links
            df_non_pareto = df_non_pareto.loc[:, ['canonical_leader', 'canonical_member', 'agent_verified', 'item_uuid', 'item_name', 'product_name']]
            df_links = df_links.loc[:, ['canonical_leader', 'canonical_member', 'agent_verified', 'item_uuid', 'item_name', 'product_name']]
            
            df_applicants = pd.concat([df_non_pareto, df_links], axis=0).reset_index(drop=True)
            df_applicants = df_applicants.drop_duplicates().reset_index(drop=True)
            df_applicants['canonical_id'] = df_applicants['canonical_leader'].map(candidate_name_id_dict)

            # re-organizing
            df_applicants = df_applicants.loc[:, ['item_uuid', 'item_name', 'canonical_id', 'canonical_leader', 'canonical_member', 'agent_verified']]
            new_links_df = df_applicants.copy()

        else:
            df_links['canonical_id'] = df_links['canonical_leader'].map(candidate_name_id_dict)
            df_links = df_links.loc[:, ['member', 'canonical_id', 'canonical_leader', 'agent_verified']]

            # mapping members back to item_uuid: | item_uuid | item_name | canonical_id | canonical_leader | canonical_member |
            new_links_df = map_member_to_item_uuid(df_links)

        # better format to dataframes
        df_canonical_candidate, new_links_df = standardize_format(df_canonical_candidate, new_links_df)

        # fixing nan values on canonical ID or canonical leader
        new_links_df = fixes_nan_on_canonical_links(new_links_df)

        # as fixing nan's on links may lead to "new" canonical items, we must verify / add to canonical
        df_canonical_candidate, new_links_df = add_missing_links_to_canonical(df_canonical_candidate, new_links_df)

        # saving datasets
        print('Saving canonical files..')
        df_canonical_candidate.to_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv', index=False)
        new_links_df.to_csv(f'canonical_data/{country}/{country}_canonical_links.csv', index=False)
        
    print('Success!')


if __name__ == "__main__":
    main()
