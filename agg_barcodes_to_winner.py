
import pandas as pd
import numpy as np

path_to_relations = '~/Downloads/[UCPC Duplicates] US PLUs - Validated TRUE relations'
path_to_barcodes = '~/Downloads/us_plus_barcodes'


def extending_barcodes(df_temp):
    # extending gtins and gtin type
    df_temp['split_gtins'] = df_temp['gtins'].str.split(',')
    df_temp['split_gtin_types'] = df_temp['gtin_types'].str.split(',')
    # transformation to rows
    df_temp_exploded = df_temp.explode(['split_gtins', 'split_gtin_types'], ignore_index=True)
    df_temp_exploded.drop(['gtins', 'gtin_types'], axis=1, inplace=True)
    df_temp_exploded.rename(columns={'split_gtins': 'gtins', 'split_gtin_types': 'gtin_types'}, inplace=True)
    df_temp_exploded = df_temp_exploded.drop_duplicates().reset_index(drop=True)
    return df_temp_exploded

def reading_files():
    print('Reading input files..')

    # winner to loser relations
    df_relations = pd.read_csv(f'{path_to_relations}.csv')

    # removing losers assigned to > 1 winner (edge case: need to fix in the heuristic - generated from equal name function)
    print(f"# losers assigned to >1 winner: {df_relations[df_relations.duplicated(['loser_entity_uuid'], keep=False)].reset_index(drop=True).shape[0]}")
    df_relations = df_relations[~df_relations.duplicated(['loser_entity_uuid'], keep=False)].reset_index(drop=True)

    # barcodes
    df_barcodes = pd.read_csv(f'{path_to_barcodes}.csv', dtype={'gtins': str})

    # flagging entities having PLU/GTIN
    df_entity_gtin_type = df_barcodes.loc[:, ['entity_uuid', 'gtin_types']]
    df_entity_gtin_type['has_plu'] = np.where(df_entity_gtin_type['gtin_types'].str.contains('PRODUCT_IDENTIFIER_TYPE_PLU'), 1, 0)
    df_entity_gtin_type['has_gtin'] = np.where(df_entity_gtin_type['gtin_types'].str.contains('PRODUCT_IDENTIFIER_TYPE_GTIN'), 1, 0)

    # fixing entities having both: GTIN & PLU (when applies)
    entities_with_gtin_and_plu = list(set(df_entity_gtin_type[(df_entity_gtin_type['has_plu'] == 1)&(df_entity_gtin_type['has_gtin'] == 1)]['entity_uuid']))
    if len(entities_with_gtin_and_plu) > 0:
        df_remove_gtin = df_barcodes[df_barcodes['entity_uuid'].isin(entities_with_gtin_and_plu)].reset_index(drop=True)
        df_barcodes = df_barcodes[~df_barcodes['entity_uuid'].isin(entities_with_gtin_and_plu)].reset_index(drop=True)

        # transforming to rows gtins stored as arrays
        df_remove_gtin = extending_barcodes(df_remove_gtin)
        print(f'# GTINs to remove: {df_remove_gtin[df_remove_gtin["gtin_types"] != "PRODUCT_IDENTIFIER_TYPE_GTIN"].shape[0]}')
        df_remove_gtin = df_remove_gtin[df_remove_gtin['gtin_types'] != 'PRODUCT_IDENTIFIER_TYPE_GTIN'].reset_index(drop=True)

        # return to array structure
        df_grouped_removed_gtins = df_remove_gtin.groupby(['entity_uuid'], dropna=False)['gtins'].agg(lambda x: ','.join(filter(None, x))).reset_index()
        df_grouped_removed_types = df_remove_gtin.groupby(['entity_uuid'], dropna=False)['gtin_types'].agg(lambda x: ','.join(filter(None, x))).reset_index()
        df_remove_back = df_grouped_removed_gtins.merge(df_grouped_removed_types, how='inner', on='entity_uuid')

        # merging to global barcodes set
        df_barcodes = pd.concat([df_barcodes, df_remove_back], axis=0).reset_index(drop=True)
        df_entity_gtin_type.loc[(df_entity_gtin_type['has_plu'] == 1)&(df_entity_gtin_type['has_gtin'] == 1), 'has_gtin'] = 0

    # resources
    df_entity_barcodes_map = df_barcodes.loc[:, ['entity_uuid', 'gtins']].drop_duplicates().reset_index(drop=True)
    df_entity_gtin_type = df_entity_gtin_type.groupby('entity_uuid').agg({'has_plu': sum, 'has_gtin': sum}).reset_index()

    # Business Rule 1: removing relations: PLU vs PLU
    df_relations = df_relations.merge(df_entity_gtin_type.rename(columns={'entity_uuid': 'winner_entity_uuid', 'has_plu': 'winner_has_plu', 'has_gtin': 'winner_has_gtin'}), how='left', on='winner_entity_uuid')
    df_relations = df_relations.merge(df_entity_gtin_type.rename(columns={'entity_uuid': 'loser_entity_uuid', 'has_plu': 'loser_has_plu', 'has_gtin': 'loser_has_gtin'}), how='left', on='loser_entity_uuid')
    print(f'# PLU vs PLU relations to remove: {df_relations[(df_relations["winner_has_plu"] == 1)&(df_relations["loser_has_plu"] == 1)].shape[0]}')
    print(f'# relations kept after cleaning: {df_relations.shape[0]}')
    df_relations = df_relations[~((df_relations["winner_has_plu"] == 1)&(df_relations["loser_has_plu"] == 1))].reset_index(drop=True)

    # saving relations as there were changes because of the business rule applied before
    df_relations.loc[:, ['winner_entity_uuid', 'loser_entity_uuid']].to_csv(f'{path_to_relations}.csv', index=False)

    # dictionary to replace losers with winners (entities) -> barcode agg
    loser_to_winner_dict = dict(zip(df_relations['loser_entity_uuid'], df_relations['winner_entity_uuid']))

    # replacing losers entity UUIDs by the winner entity UUID to which they are related (to map gtin type)
    df_barcodes['entity_uuid'] = df_barcodes['entity_uuid'].replace(loser_to_winner_dict)

    # extending barcodes set
    df_barcodes_exploded = extending_barcodes(df_barcodes)

    return df_relations, df_entity_barcodes_map, df_barcodes_exploded

def join_barcodes(row):
        values = [str(row['winner_gtins']), str(row['loser_gtins'])]
        non_null_values = [value for value in values if value != 'None']
        return ', '.join(set(non_null_values)) if non_null_values else None

def mapping_barcodes_and_building_arrays(df_relations, df_entity_barcodes_map):
    print('Mapping barcodes to winners and losers..')

    # mapping barcodes to winners and losers
    df_relations = df_relations.merge(df_entity_barcodes_map.rename(columns={'entity_uuid': 'winner_entity_uuid', 'gtins': 'winner_gtins'}), how='left', on='winner_entity_uuid')
    df_relations = df_relations.merge(df_entity_barcodes_map.rename(columns={'entity_uuid': 'loser_entity_uuid', 'gtins': 'loser_gtins'}), how='left', on='loser_entity_uuid')
    df_relations = df_relations.sort_values('winner_entity_uuid').reset_index(drop=True)
    
    # removing aggregations between PLUs and GTINs
    print(f'# winner is PLU vs loser is GTIN: {df_relations[(df_relations["winner_has_plu"] == 1)&(df_relations["loser_has_gtin"] == 1)].shape[0]}')
    print(f'# winner is GTIN vs loser is PLU: {df_relations[(df_relations["winner_has_gtin"] == 1)&(df_relations["loser_has_plu"] == 1)].shape[0]}')
    df_relations.loc[(df_relations['winner_has_plu'] == 1)&(df_relations['loser_has_gtin'] == 1), 'loser_gtins'] = np.nan
    df_relations.loc[(df_relations['winner_has_plu'] == 1)&(df_relations['loser_has_gtin'] == 1), 'loser_has_gtin'] = 0
    df_relations.loc[(df_relations['winner_has_gtin'] == 1)&(df_relations['loser_has_plu'] == 1), 'winner_gtins'] = np.nan
    df_relations.loc[(df_relations['winner_has_gtin'] == 1)&(df_relations['loser_has_plu'] == 1), 'winner_has_gtin'] = 0

    # grouping losers barcodes
    df_loser_gtins = df_relations[~df_relations['loser_gtins'].isna()][['winner_entity_uuid', 'loser_gtins']].reset_index(drop=True)
    df_grouped_loser_gtins = df_loser_gtins.groupby(['winner_entity_uuid'], dropna=False)['loser_gtins'].agg(lambda x: ','.join(filter(None, set(x)))).reset_index()

    # grouping winner barcodes
    df_winner_gtins = df_relations[~df_relations['winner_gtins'].isna()][['winner_entity_uuid', 'winner_gtins']].reset_index(drop=True)
    df_grouped_winner_gtins = df_winner_gtins.groupby(['winner_entity_uuid'], dropna=False)['winner_gtins'].agg(lambda x: ','.join(filter(None, set(x)))).reset_index()

    # merging all
    df = df_relations.loc[:, ['winner_entity_uuid']].drop_duplicates().reset_index(drop=True)
    df = df.merge(df_grouped_winner_gtins, how='left', on='winner_entity_uuid')
    df = df.merge(df_grouped_loser_gtins, how='left', on='winner_entity_uuid')

    # union of columns
    df = df.replace(np.nan, None)
    df['gtins'] = df[['winner_gtins', 'loser_gtins']].apply(lambda row: join_barcodes(row), axis=1)
    df = df.loc[:, ['winner_entity_uuid', 'gtins']].reset_index(drop=True)

    return df

def transforming_into_bulk(df, df_barcodes_exploded):
    print('Transforming into bulk update format..')

    # transforming into rows
    df['split_gtins'] = df['gtins'].str.split(',')
    df_exploded = df.explode('split_gtins', ignore_index=True)
    df_exploded.drop('gtins', axis=1, inplace=True)

    # removing nulls and duplicated gtins per entity
    df_exploded = df_exploded[~df_exploded['split_gtins'].isna()].reset_index(drop=True)
    df_exploded = df_exploded.drop_duplicates().reset_index(drop=True)
    
    # adding columns for bulk action
    df_exploded['CatalogUUID'] = 'a2781468-c663-4224-a62f-596bc293e5ac'
    df_exploded.rename(columns={'winner_entity_uuid': 'ProductUUID', 'split_gtins': 'GlobalIdentifier'}, inplace=True)

    # adding gtin type
    df_barcodes_exploded.rename(columns={'entity_uuid': 'ProductUUID', 'gtins': 'GlobalIdentifier'}, inplace=True)

    df_final = df_exploded.merge(df_barcodes_exploded, how='left', on=['ProductUUID', 'GlobalIdentifier'])
    df_final = df_final.sort_values('ProductUUID').reset_index(drop=True)
    df_final.rename(columns={'gtin_types': 'Type'}, inplace=True)

    # fixing null gtin type
    df_final.loc[(df_final['GlobalIdentifier'].str.len() < 7)&(df_final['Type'].isna()), 'Type'] = 'PRODUCT_IDENTIFIER_TYPE_PLU'
    df_final.loc[(df_final['GlobalIdentifier'].str.len() >= 7)&(df_final['Type'].isna()), 'Type'] = 'PRODUCT_IDENTIFIER_TYPE_GTIN'

    # transforming gtin type
    gtin_type_dict = {'PRODUCT_IDENTIFIER_TYPE_GTIN': 'GTIN', 'PRODUCT_IDENTIFIER_TYPE_PLU': 'PLU'}
    df_final['Type'] = df_final['Type'].map(gtin_type_dict)

    # organizing
    df_final = df_final.loc[:, ['CatalogUUID', 'ProductUUID', 'GlobalIdentifier', 'Type']]
    df_final = df_final.sort_values('ProductUUID').reset_index(drop=True)
    df_final = df_final.drop_duplicates(['ProductUUID', 'GlobalIdentifier']).reset_index(drop=True)

    # override
    df_final = df_final.sort_values('ProductUUID').reset_index(drop=True)
    df_final['previous_uuid'] = df_final['ProductUUID'].shift(1)
    df_final['Override'] = np.where(df_final['ProductUUID'] == df_final['previous_uuid'], "No", "Yes")
    df_final.drop('previous_uuid', axis=1, inplace=True)

    return df_final


def main():

    # reading input files
    df_relations, df_entity_barcodes_map, df_barcodes_exploded = reading_files()

    # mapping barcodes to winners and losers
    df = mapping_barcodes_and_building_arrays(df_relations, df_entity_barcodes_map)

    # transforming into bulk update format
    df_final = transforming_into_bulk(df, df_barcodes_exploded)

    # saving to run in the bulk tool
    df_final.to_csv(f'~/Downloads/barcode_update_on_winner.csv', index=False)


if __name__ == "__main__":
    main()
