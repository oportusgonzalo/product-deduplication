
import pandas as pd


country = ''
parent_chain = ''


def main():
    
    # reading files
    df_canonical = pd.read_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv')
    df_new = pd.read_csv(f'missing/{country}/{country}_canonical_catalog.csv')

    # verifying old leaders lost in the concatenation
    canonical_leaders_new = list(set(df_new['canonical_leader']))
    print(f'N° old canonical leaders lost: {df_canonical[~(df_canonical["canonical_leader"].isin(canonical_leaders_new))].reset_index(drop=True).shape[0]}')

    # modified ids
    canonical_leaders_old_ids = dict(zip(df_canonical['canonical_leader'], df_canonical['canonical_id']))
    i = 0
    for leader_, id_ in canonical_leaders_old_ids.items():
       if df_new[df_new['canonical_leader'] == leader_]['canonical_id'].values != id_:
            i += 1
    print(f'N° old leaders with modified id: {i}')

    # verifying all missing products were linked
    df_missing = pd.read_csv(f'missing/{country}/{country}_missing_{parent_chain}.csv')
    df_new_links = pd.read_csv(f'missing/{country}/{country}_canonical_links.csv')
    linked_item_names = list(set(df_new_links['item_name']))
    linked_item_uuids = list(set(df_new_links['item_uuid']))
    print(f'N° missing item names not linked: {df_missing[~(df_missing["item_name"].isin(linked_item_names))].reset_index(drop=True).shape[0]}')
    print(f'N° missing item uuids not linked: {df_missing[~(df_missing["item_uuid"].isin(linked_item_uuids))].reset_index(drop=True).shape[0]}')


if __name__ == "__main__":
    main()