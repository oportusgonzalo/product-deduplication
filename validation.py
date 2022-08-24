
import pandas as pd


stores_to_review = ['booker', 'nisa']

def read_back_propagation_file(store):
    return pd.read_csv(f'back_propagation/raw_vs_clean_uk_{store}_products_85_75.csv')


def main():
    # reading canonical links file
    canonical_links_df = pd.read_csv('canonical_data/canonical_links.csv')

    df_raw = pd.DataFrame()
    for store in stores_to_review:
        df_temp = read_back_propagation_file(store)
        df_raw = pd.concat([df_raw, df_temp], axis=0).reset_index(drop=True)

    # list of all unique UUIDs
    raw_items = list(set(df_raw['item_uuid']))

    print(f'Number of item UUIDs not added to links file: {len(list(set(canonical_links_df[~canonical_links_df["item_uuid"].isin(raw_items)]["item_uuid"])))}')
    


if __name__ == "__main__":
    main()