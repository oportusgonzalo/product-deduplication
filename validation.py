
import pandas as pd



def read_back_propagation_file(store, country):
    return pd.read_csv(f'back_propagation/{country}/raw_vs_clean_{country}_{store}_products_85_75.csv')

def number_uuids_involved(df_raw):
    print(f'Number of unique UUIDs processed: {len(list(set(df_raw["item_uuid"])))}')
    print(f'Number of unique item names processed: {len(list(set(df_raw["item_name"])))}')

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

def main(country, country_stores_dict):
    print(f'Validation statistics for {country.title()} coming up..')

    df_raw = pd.DataFrame()
    for store in country_stores_dict[country]:
        df_temp = read_back_propagation_file(store, country)
        df_raw = pd.concat([df_raw, df_temp], axis=0).reset_index(drop=True)
    
    # reading canonical links file
    canonical_links_df = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')

    # total number of unique UUIDs processed
    number_uuids_involved(df_raw)

    # number of UUIDs not linked
    uuids_not_added(canonical_links_df, df_raw)

    # item name decline --> unique percentage of leaders
    decline_items_number(canonical_links_df, df_raw)

    # products that haven't been verified by agents
    agent_unverified(canonical_links_df)


if __name__ == "__main__":
    
    country_stores_dict = {
        'uk': ['booker', 'nisa', 'costcutter', 'bestway', 'rev_nisa', 'rev_costcutter'],
        'cr': ['dp&az', 'ampm', 'fresh_market', 'rev_dp&az', 'rev_ampm']
    }

    for country in ['uk', 'cr']:
        main(country, country_stores_dict)
        print()
