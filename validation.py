
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

def global_statistics(country):
    print(f'GLOBAL STATISTICS OF {country.upper()}')
    df_links = pd.read_csv(f'canonical_data/{country}/{country}_canonical_links.csv')
    print(f'N° raw item UUIDs: {len(list(set(df_links["item_uuid"])))}')
    print(f'N° raw item names: {len(list(set(df_links["item_name"])))}')
    print(f'N° raw items not verified by agents: {df_links[df_links["agent_verified"] == 0].drop_duplicates("item_name").shape[0]}')
    print(f'% raw items not verified by agents: {round(df_links[df_links["agent_verified"] == 0].drop_duplicates("item_name").shape[0] / len(list(set(df_links["item_name"]))), 3)}')
    print(f'N° canonical IDs: {len(list(set(df_links["canonical_id"])))}')
    print(f'% canonical IDs / Total: {round(len(list(set(df_links["canonical_id"]))) / len(list(set(df_links["item_name"]))), 3)}')

def main(country, country_stores_dict):
    print(f'Validation statistics for {country.upper()} coming up..')

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

    # global statistics for report
    print()
    global_statistics(country)


if __name__ == "__main__":
    
    country_stores_dict = {
        'uk': ['booker', 'nisa', 'costcutter', 'bestway', 'rev_nisa', 'rev_costcutter', 'rev_bestway', 'rev_booker', 'keresley_news', 'tulsi_news', 'all_in_hornchurch', 'cricklewood', 'montrose_street', 'fmcg1', 'reckitt'],
        'cr': ['dp&az', 'ampm', 'fresh_market', 'rev_dp&az', 'rev_ampm', 'rev_fresh_market', 'farmavalue', 'fciabombacr', 'dos_pinos_cr', 'musicr', 'peqmundo'],
        'za': ['brands1', 'brands2', 'pepsico'],
        'us&ca': ['pepsico', 'unilever']
    }

    for country in ['uk', 'cr', 'za', 'us&ca']:
        main(country, country_stores_dict)
        print()
