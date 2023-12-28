
import pandas as pd


path_to_heuristic_output = '~/Downloads/heuristic_duplicates/95_output_br_ucpc_duplicates'
path_to_score_input = '~/Downloads/ucpc_score_input'

'''STEPS:
1. For each group, verify which has the highest score
2. Save the highest score entity in a different column (new winner)
'''


def reading_files():
    print('Reading heuristic output & score input..')

    # heuristic output
    df_output = pd.read_csv(f'{path_to_heuristic_output}.csv')
    df_winner = df_output.loc[:, ['winner_entity_uuid', 'winner_name']].rename(columns={'winner_entity_uuid': 'entity_uuid', 'winner_name': 'name'})
    df_loser = df_output.loc[:, ['loser_entity_uuid', 'loser_name']].rename(columns={'loser_entity_uuid': 'entity_uuid', 'loser_name': 'name'})
    df_output.drop(['winner_name', 'loser_name'], axis=1, inplace=True)
    df_entity_name = pd.concat([df_winner, df_loser], axis=0).reset_index(drop=True)
    df_entity_name = df_entity_name.drop_duplicates().reset_index(drop=True)

    # score input
    df_score = pd.read_csv(f'{path_to_score_input}.csv')

    return df_output, df_entity_name, df_score

def calculating_score(df_score):
    print('Calculating a score for each entity: completeness, # linked, & scans')

    # assigning a score to each of the 14 attributes: the sum equals 1 (ordered by importance)
    name_score = 0.12
    image_score = 0.12
    upt_score = 0.1
    gtin_score = 0.1
    net_qty_score = 0.09
    pack_size_score = 0.08
    func_name_score = 0.07
    brand_score = 0.07
    container_score = 0.07
    weight_score = 0.06
    dimensions_score = 0.05
    abv_score = 0.03
    description_score = 0.03
    vintage_score = 0.01

    # dictionary to multiply each column by the score
    score_calculation_dict = {
        'has_name': name_score, 
        'has_func_name': func_name_score, 
        'has_brand': brand_score, 
        'has_image': image_score,
       'has_gtin': gtin_score, 
       'has_upt': upt_score, 
       'has_dimensions': dimensions_score, 
       'has_pack_size': pack_size_score,
       'has_container': container_score, 
       'has_net_qty': net_qty_score, 
       'has_weight': weight_score, 
       'has_description': description_score,
       'has_abv': abv_score, 
       'has_vintage': vintage_score
    }

    # attribute individual score
    for col_, score_ in score_calculation_dict.items():
        df_score[col_] = df_score[col_] * score_

    # calculating completeness score & removing base columns
    df_score['att_score'] = df_score[score_calculation_dict.keys()].sum(axis=1)
    df_score.drop(score_calculation_dict.keys(), axis=1, inplace=True)

    # incorrect scan rate
    df_score.loc[df_score['scans'] > 3, 'inc_scan_rate'] = round(df_score[df_score['scans'] > 3]['incorrect_scans'] / df_score[df_score['scans'] > 3]['scans'], 2)
    df_score.loc[df_score['scans'] <= 3, 'inc_scan_rate'] = 0
    df_score.drop(['scans', 'correct_scans', 'incorrect_scans'], axis=1, inplace=True)

    # z-score for number linked products: measures how manu std an entity's # of linked products is from the mean
    mean_value = df_score['linked_products'].mean()
    std_dev = df_score['linked_products'].std()
    df_score['z_score'] = df_score['linked_products'].apply(lambda x: round((x-mean_value) / std_dev, 2))
    df_score.drop('linked_products', axis=1, inplace=True)

    # calculating a global score
    '''Weights will be assigned to each measure, they are non-equal in importance: (att_score = 0.6), (inc_scan_rate = 0.2), (z_score = 0.2)'''
    df_score['score'] = (df_score['att_score'] * 0.6) + (df_score['inc_scan_rate'] * 0.2) + (df_score['z_score'] * 0.2)
    df_score.drop(['att_score', 'inc_scan_rate', 'z_score'], axis=1, inplace=True)

    return df_score

def verifies_highest_score_entity(df_output, df_score):
    print('For each duped group: verifies which entity has the highest score..')

    df_groups = pd.DataFrame(columns=['winner_entity_uuid', 'loser_entity_uuid'])
    # iterating over each group: leaders lead
    for winner_ in df_output['winner_entity_uuid'].unique(): 
        df_temp = df_output[df_output['winner_entity_uuid'] == winner_].reset_index(drop=True)

        # concatenate winner & loser entities into a single column to merge scores
        combined_series = pd.concat([df_temp['winner_entity_uuid'], df_temp['loser_entity_uuid']], axis=0, ignore_index=True)
        df_combined = pd.DataFrame(combined_series, columns=['entity_uuid'])
        df_combined = df_combined.drop_duplicates().reset_index(drop=True)
        df_combined['winner?'] = 0
        df_combined.loc[0, 'winner?'] = 1

        # merging scores
        df_combined = df_combined.merge(df_score, how='left', on='entity_uuid')
        df_combined.loc[df_combined['score'].isna(), 'score'] = 0

        # identify the max score
        max_index = df_combined['score'].idxmax()
        df_combined['max_score_entity'] = df_combined.loc[max_index, 'entity_uuid']

        # re-grouping with new leader and concatenating to global groups DF
        df_new_grouping = df_combined.drop(['winner?', 'score'], axis=1).rename(columns={'max_score_entity': 'winner_entity_uuid', 'entity_uuid': 'loser_entity_uuid'})
        df_new_grouping = df_new_grouping.loc[:, ['winner_entity_uuid', 'loser_entity_uuid']]
        df_new_grouping = df_new_grouping[df_new_grouping['winner_entity_uuid'] != df_new_grouping['loser_entity_uuid']].reset_index(drop=True)
        df_groups = pd.concat([df_groups, df_new_grouping], axis=0).reset_index(drop=True)

    return df_groups

def additional_data(df_groups, df_entity_name):
    print('Adding product names as for agents to review..')

    # adding winner product name
    df_groups = df_groups.merge(df_entity_name, how='left', left_on='winner_entity_uuid', right_on='entity_uuid')
    df_groups.rename(columns={'name': 'winner_name'}, inplace=True)
    df_groups.drop('entity_uuid', axis=1, inplace=True)

    # adding loser product name
    df_groups = df_groups.merge(df_entity_name, how='left', left_on='loser_entity_uuid', right_on='entity_uuid')
    df_groups.rename(columns={'name': 'loser_name'}, inplace=True)
    df_groups.drop('entity_uuid', axis=1, inplace=True)

    # organizing
    df_groups = df_groups.loc[:, ['winner_entity_uuid', 'winner_name', 'loser_entity_uuid', 'loser_name']]

    return df_groups


def main():

    # reading heuristic output
    df_output, df_entity_name, df_score = reading_files()

    # calculating a score for each entity: completeness, # linked, & scans
    df_score = calculating_score(df_score)

    # for each duped group: verifies which entity has the highest score
    df_groups = verifies_highest_score_entity(df_output, df_score)

    # adding product names as for agents to review
    df_groups = additional_data(df_groups, df_entity_name)

    # saving result
    df_groups.to_csv(f'{path_to_score_input}_strategic_winner.csv', index=False)


if __name__ == '__main__':
    main()
