
import pandas as pd
import numpy as np

path_to_relations = '~/Downloads/[UCPC Duplicates] US PLUs - Dups'
path_to_barcodes = '~/Downloads/us_plus_barcodes'


def reading_files():
    print('Reading input files..')

    # winner to loser relations
    df_relations = pd.read_csv(f'{path_to_relations}.csv')

    # removing losers assigned to > 1 winner (edge case: need to fix in the heuristic - generated from equal name function)
    print(f"# losers assigned to >1 winner: {df_relations[df_relations.duplicated(['loser_entity_uuid'], keep=False)].reset_index(drop=True).shape[0]}")
    df_relations = df_relations[~df_relations.duplicated(['loser_entity_uuid'], keep=False)].reset_index(drop=True)
    loser_to_winner_dict = dict(zip(df_relations['loser_entity_uuid'], df_relations['winner_entity_uuid']))

    # barcodes
    df_barcodes = pd.read_csv(f'{path_to_barcodes}.csv', dtype={'gtins': str})
    entity_barcodes_dict = dict(zip(df_barcodes['entity_uuid'], df_barcodes['gtins']))

    # replacing losers entity UUIDs by the winner entity UUID to which they are related (to map gtin type)
    df_barcodes['entity_uuid'] = df_barcodes['entity_uuid'].replace(loser_to_winner_dict)

    # extending barcodes set
    df_barcodes['split_gtins'] = df_barcodes['gtins'].str.split(',')
    df_barcodes['split_gtin_types'] = df_barcodes['gtin_types'].str.split(',')
    df_barcodes_exploded = df_barcodes.explode(['split_gtins', 'split_gtin_types'], ignore_index=True)
    df_barcodes_exploded.drop(['gtins', 'gtin_types'], axis=1, inplace=True)
    df_barcodes_exploded.rename(columns={'split_gtins': 'gtins', 'split_gtin_types': 'gtin_types'}, inplace=True)
    df_barcodes_exploded = df_barcodes_exploded.drop_duplicates().reset_index(drop=True)

    return df_relations, entity_barcodes_dict, df_barcodes_exploded

def mapping_barcodes_and_building_arrays(df_relations, entity_barcodes_dict):
    print('Mapping barcodes to winners and losers..')

    # mapping
    df_relations['winner_gtins'] = df_relations['winner_entity_uuid'].map(entity_barcodes_dict)
    df_relations['loser_gtins'] = df_relations['loser_entity_uuid'].map(entity_barcodes_dict)

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
    df['gtins'] = df['winner_gtins'] + ',' + df['loser_gtins']
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
    df_final.rename(columns={'gtin_types': 'Type'}, inplace=True)

    # organizing
    df_final = df_final.loc[:, ['CatalogUUID', 'ProductUUID', 'GlobalIdentifier', 'Type']]

    # override
    df_final = df_final.sort_values('ProductUUID').reset_index(drop=True)
    df_final['previous_uuid'] = df_final['ProductUUID'].shift(1)
    df_final['Override'] = np.where(df_final['ProductUUID'] == df_final['previous_uuid'], "No", "Yes")
    df_final.drop('previous_uuid', axis=1, inplace=True)

    return df_final


def main():

    # reading input files
    df_relations, entity_barcodes_dict, df_barcodes_exploded = reading_files()

    # mapping barcodes to winners and losers
    df = mapping_barcodes_and_building_arrays(df_relations, entity_barcodes_dict)

    # transforming into bulk update format
    df_final = transforming_into_bulk(df, df_barcodes_exploded)

    # saving to run in the bulk tool
    df_final.to_csv(f'~/Downloads/barcode_update_on_winner.csv', index=False)


if __name__ == "__main__":
    main()
