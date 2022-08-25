
import pandas as pd


stores_to_review = ['booker', 'nisa']

def read_back_propagation_file(store):
    return pd.read_csv(f'back_propagation/raw_vs_clean_uk_{store}_products_85_75.csv')

def number_uuids_involved(df_raw):
    print(f'Number of unique UUIDs processed: {len(list(set(df_raw["item_uuid"])))}')

def uuids_not_added(canonical_links_df, df_raw):
    # list of all unique UUIDs
    raw_items = list(set(df_raw['item_uuid']))
    print(f'Number of UUIDs not added to links file: {len(list(set(canonical_links_df[~canonical_links_df["item_uuid"].isin(raw_items)]["item_uuid"])))}')

def decline_items_number(canonical_links_df, df_raw):
    n_raw_items = len(list(set(df_raw['item_name'])))
    n_canonical_leaders = len(list(set(canonical_links_df['canonical_leader'])))
    print(f'Percentage of decline in the number of unique item names - canonical leaders: {round(n_canonical_leaders/n_raw_items, 3)}')

def agent_unverified(canonical_links_df):
    not_verified_df = canonical_links_df[canonical_links_df['agent_verified'] == 0].drop_duplicates().reset_index(drop=True)
    print(f'Number of products not verified by agents: {not_verified_df.shape[0]}')
    print(f'Percentage of products not verified by agents: {round(not_verified_df.shape[0]/canonical_links_df.shape[0], 3)}')

def main():
    print('Validation statistics coming up..')
    # reading canonical links file
    canonical_links_df = pd.read_csv('canonical_data/canonical_links.csv')

    df_raw = pd.DataFrame()
    for store in stores_to_review:
        df_temp = read_back_propagation_file(store)
        df_raw = pd.concat([df_raw, df_temp], axis=0).reset_index(drop=True)

    # total number of unique UUIDs processed
    number_uuids_involved(df_raw)

    # number of UUIDs not linked
    uuids_not_added(canonical_links_df, df_raw)

    # item name decline --> unique percentage of leaders
    decline_items_number(canonical_links_df, df_raw)

    # products that haven't been verified by agents
    agent_unverified(canonical_links_df)
    

if __name__ == "__main__":
    main()