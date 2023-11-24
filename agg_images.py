
import pandas as pd
import numpy as np


path_to_true_relations = '~/Downloads/[UCPC Duplicates] US PLUs - Validated TRUE relations'
path_to_images = '~/Downloads/ucpc_images'

def read_files():
    print('Reading files..')
    
    df_relations = pd.read_csv(f'{path_to_true_relations}.csv', usecols=['winner_entity_uuid', 'loser_entity_uuid'])
    df_images = pd.read_csv(f'{path_to_images}.csv')
    df_images = df_images.drop_duplicates().reset_index(drop=True)

    return df_relations, df_images

def winners_needing_an_update(df_relations, df_images):
    print('Identifying winners needing an update: no image or image not 3P..')

    # merging winner images
    df_winner = df_relations.merge(df_images, how='left', left_on='winner_entity_uuid', right_on='ucpc_entity_uuid')
    df_winner.drop(['loser_entity_uuid', 'ucpc_entity_uuid'], axis=1, inplace=True)
    df_winner = df_winner.drop_duplicates().reset_index(drop=True)

    # melting
    df_winner = pd.melt(df_winner, id_vars=['winner_entity_uuid'], var_name='variable', value_name='value')

    # splitting images from sources
    df_winner_images = df_winner[~df_winner['variable'].str.contains('image_source')].reset_index(drop=True)
    df_winner_sources = df_winner[df_winner['variable'].str.contains('image_source')].reset_index(drop=True)
    df_winner_sources['variable'] = df_winner_sources['variable'].apply(lambda x: x.replace('_source', ''))

    # merging to get the structure: entity | image | source
    df_winner_merge = df_winner_images.merge(df_winner_sources, how='left', on=['winner_entity_uuid', 'variable'])
    df_winner_merge = df_winner_merge[(~df_winner_merge['value_x'].isna())&(df_winner_merge['value_x'] != r'\N')].reset_index(drop=True)
    df_winner_merge.rename(columns={'value_x': 'image_url', 'value_y': 'image_source'}, inplace=True)

    # flagging images as 3P
    df_winner_merge['3p_image'] = np.where(df_winner_merge['image_source'] == 'IMAGE_SOURCE_THIRD_PARTY_APPROVED', 1, 0)

    # groupby winner and sum 3P images
    df_winner_3p = df_winner_merge.groupby('winner_entity_uuid').agg({'3p_image': sum}).reset_index()

    # winners to update: either doesn't have an image or image isn't 3P
    df_winners_flagged = df_relations.loc[:, ['winner_entity_uuid']].drop_duplicates().reset_index(drop=True)
    df_winners_flagged = df_winners_flagged.merge(df_winner_3p, how='left', on='winner_entity_uuid')
    df_winners_flagged = df_winners_flagged[df_winners_flagged['3p_image'] != 1].reset_index(drop=True)
    winners_can_be_updated_list = list(set(df_winners_flagged['winner_entity_uuid']))

    print(f'# winners are candidates to have the image updated: {len(winners_can_be_updated_list)}')

    # flagging winner results
    df_winners_flagged.loc[df_winners_flagged['3p_image'].isna(), 'winner_image'] = 'No Image'
    df_winners_flagged.loc[df_winners_flagged['3p_image'] == 0, 'winner_image'] = 'Image not 3p'
    df_winners_flagged.drop('3p_image', axis=1, inplace=True)
    
    return df_winners_flagged, winners_can_be_updated_list    

def understanding_losers_images_with_stats(df_relations, df_images, winners_can_be_updated_list):
    print('Understanding if losers have images and if they are 3p..')

    # filtering by winners that could be having an image update
    df_filter = df_relations[df_relations['winner_entity_uuid'].isin(winners_can_be_updated_list)].reset_index(drop=True)

    # adding losers images
    df_filter_with_loser_images = df_filter.merge(df_images, how='left', left_on='loser_entity_uuid', right_on='ucpc_entity_uuid')
    df_filter_with_loser_images.drop('ucpc_entity_uuid', axis=1, inplace=True)

    # adapting
    df_filter_with_loser_images_melted = pd.melt(df_filter_with_loser_images, id_vars=['winner_entity_uuid', 'loser_entity_uuid'], var_name='variable', value_name='value')
    df_filter_images = df_filter_with_loser_images_melted[~df_filter_with_loser_images_melted['variable'].str.contains('image_source')].reset_index(drop=True)
    df_filter_sources = df_filter_with_loser_images_melted[df_filter_with_loser_images_melted['variable'].str.contains('image_source')].reset_index(drop=True)
    df_filter_sources['variable'] = df_filter_sources['variable'].apply(lambda x: x.replace('_source', ''))
    df_filter_merge = df_filter_images.merge(df_filter_sources, how='left', on=['winner_entity_uuid', 'loser_entity_uuid', 'variable'])
    df_filter_merge = df_filter_merge[(~df_filter_merge['value_x'].isna())&(df_filter_merge['value_x'] != r'\N')].reset_index(drop=True)
    df_filter_merge.rename(columns={'value_x': 'image_url', 'value_y': 'image_source'}, inplace=True)

    # flagging according to losers data
    df_filter_merge['loser_has_image'] = np.where(df_filter_merge['image_url'] != np.nan, 1, 0)
    df_filter_merge['loser_3p_image'] = np.where(df_filter_merge['image_source'] == 'IMAGE_SOURCE_THIRD_PARTY_APPROVED', 1, 0)

    # stats
    df_loser_stats = df_filter_merge.groupby(['winner_entity_uuid', 'loser_entity_uuid']).agg({'loser_has_image': sum, 'loser_3p_image': sum}).reset_index() 

    return df_filter_merge, df_loser_stats

def mapping_losers_images_to_winner(df_winners_flagged, df_filter_merge, df_loser_stats):
    print('Mapping loser images to winner..')

    # adding winner stats to loser stats
    df_stats = df_loser_stats.merge(df_winners_flagged, how='left', on='winner_entity_uuid')

    # case 1: winner can be enriched by loser having 3P image
    df_win_to_3p_update = df_stats[df_stats['loser_3p_image'] == 1].drop(['loser_has_image', 'loser_3p_image', 'winner_image'], axis=1).reset_index(drop=True)
    winners_to_3p_list = list(set(df_win_to_3p_update['winner_entity_uuid']))
    df_stats = df_stats[~df_stats['winner_entity_uuid'].isin(winners_to_3p_list)].reset_index(drop=True)
    df_winners_flagged.loc[df_winners_flagged['winner_entity_uuid'].isin(winners_to_3p_list), 'result'] = '3p from loser'
    print(f"# winners that couldn't be mapped to 3P image: {df_winners_flagged[df_winners_flagged['result'].isna()].shape[0]}")

    # case 2: winner doesn't have an image but a loser has
    df_win_to_image = df_stats[(df_stats['loser_has_image'] == 1)&(df_stats['winner_image'] == 'No Image')].drop(['loser_has_image', 'loser_3p_image', 'winner_image'], axis=1).reset_index(drop=True)
    win_to_image_list = list(set(df_win_to_image['winner_entity_uuid']))
    df_stats = df_stats[~df_stats['winner_entity_uuid'].isin(win_to_image_list)].reset_index(drop=True)
    df_winners_flagged.loc[df_winners_flagged['winner_entity_uuid'].isin(win_to_image_list), 'result'] = 'Adds image'
    
    print(f'# winners without image at the end of the mapping: {df_winners_flagged[(df_winners_flagged["result"].isna())&(df_winners_flagged["winner_image"] == "No Image")].shape[0]}')

    return df_winners_flagged, df_win_to_3p_update, df_win_to_image

def adding_loser_images_to_winner(df_filter_merge, df_win_to_3p_update, df_win_to_image):
    print('Adding losers images to each winner - also selecting one winner <>loser relation..')

    # removing duplicated assignments on each mapping set
    df_win_to_3p_update = df_win_to_3p_update.drop_duplicates('winner_entity_uuid').reset_index(drop=True)
    df_win_to_image = df_win_to_image.drop_duplicates('winner_entity_uuid').reset_index(drop=True)

    # concatenating both sets: 3P & no image
    df = pd.concat([df_win_to_3p_update, df_win_to_image], axis=0).reset_index(drop=True)

    # adding loser images to winners
    df_filter_merge.drop(['loser_has_image', 'loser_3p_image'], axis=1, inplace=True)
    df = df.merge(df_filter_merge, how='left', on=['winner_entity_uuid', 'loser_entity_uuid']).sort_values(['winner_entity_uuid', 'variable']).reset_index(drop=True)
    df.drop(['loser_entity_uuid', 'variable'], axis=1, inplace=True)
    print(f'# winners for some reason have no mapping of images: {df[df["image_url"].isna()].shape[0]}')
    
    return df

def bulk_format(df):
    print('Transforming into bulk action format..')

    # variables
    df['CatalogUUID'] = 'a2781468-c663-4224-a62f-596bc293e5ac'
    df['CatalogImageSource'] = None
    df.rename(columns={'winner_entity_uuid': 'ProductUUID', 'image_url': 'ImageURL', 'image_source': 'ImageSource'}, inplace=True)

    # organizing
    df = df.loc[:, ['CatalogUUID', 'ProductUUID', 'ImageURL', 'ImageSource', 'CatalogImageSource']]

    # override logic
    df['previous_uuid'] = df['ProductUUID'].shift(1)
    df['Override'] = np.where(df['ProductUUID'] == df['previous_uuid'], "No", "Yes")
    df.drop('previous_uuid', axis=1, inplace=True)

    print(f'# winners to update: {len(list(set(df["ProductUUID"])))}')

    return df


def main():

    # reading raw files
    df_relations, df_images = read_files()

    # identifying winners needing an update: no image or image not 3P
    df_winners_flagged, winners_can_be_updated_list = winners_needing_an_update(df_relations, df_images)

    # understanding if losers have images and if they are 3p
    df_filter_merge, df_loser_stats = understanding_losers_images_with_stats(df_relations, df_images, winners_can_be_updated_list)

    # mapping loser images to winner
    df_winners_flagged, df_win_to_3p_update, df_win_to_image = mapping_losers_images_to_winner(df_winners_flagged, df_filter_merge, df_loser_stats)

    # adding losers images to each winner - also selecting one winner <>loser relation
    df = adding_loser_images_to_winner(df_filter_merge, df_win_to_3p_update, df_win_to_image)

    # transforming into bulk action format
    df = bulk_format(df)

    # saving
    df.to_csv('~/Downloads/update_winners_with_losers_images.csv', index=False)


if __name__ == "__main__":
    main()
