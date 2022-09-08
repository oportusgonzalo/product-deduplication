
import pandas as pd

from static import *

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


country = ''
language_='en'
match_canonical = True


def read_uk_barcode_files():
    print('Reading barcode files..')
    df = pd.DataFrame()
    # reading files with barcode content
    df_1 = pd.read_csv(f'barcodes_input/UK NV EANs Collection v1.csv', header=1)
    df_2 = pd.read_csv(f'barcodes_input/UK NV Local Catalog Data v1.csv')
    df_1.columns = df_1.columns.str.strip().str.lower()
    df_2.columns = df_2.columns.str.strip().str.lower()

    # useful columns selection
    df_1 = df_1.loc[:, ['ean.1', 'sku_name.1']].rename(columns={'ean.1': 'ean', 'sku_name.1': 'item_name'})
    df_2 = df_2.loc[:, ['ean', 'sku_name']].rename(columns={'sku_name': 'item_name'})

    df = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
    df = df[~(df['ean'].isna()|df['item_name'].isna())].reset_index(drop=True)

    df.drop_duplicates('item_name', inplace=True)
    df = df.sort_values(by=['item_name']).reset_index(drop=True)
    
    return df

def just_one_barcode(df_na, df_latam):
    print('Randomly selecting one barcode per product..')

    # removiendo brackets
    df_na['barcodes'] = df_na['barcodes'].apply(lambda x: re.sub(r'(\{)|(\})', '', x)).astype(str)
    df_latam['barcodes'] = df_latam['barcodes'].apply(lambda x: re.sub(r'(\{)|(\})', '', x)).astype(str)

    # seleccionamos un barcode por producto (USE PANDAS EXPLODE)
    df_na['barcodes'] = df_na['barcodes'].str.split(',').str[0]
    df_latam['barcodes'] = df_latam['barcodes'].str.split(',').str[0]

    return df_na, df_latam

def read_cornershop_barcode_files():
    print('Reading Cornershop barcode files..')

    # reading files
    df_na = pd.read_csv('barcodes_input/barcodes_NA.csv')
    df_latam = pd.read_csv('barcodes_input/barcodes_LATAM.csv')
    df_na.columns = df_na.columns.str.strip().str.lower()
    df_latam.columns = df_latam.columns.str.strip().str.lower()

    # removing names with nan and selecting useful columns
    df_na = df_na[~df_na['name'].isna()].reset_index(drop=True).drop('country', axis=1)
    df_latam = df_latam[~df_latam['name'].isna()].reset_index(drop=True).drop('country', axis=1)
    df_latam = df_latam[~df_latam['barcodes'].isna()].reset_index(drop=True)

    # as products may have > 1 barcode, we randomly select 1
    df_na, df_latam = just_one_barcode(df_na, df_latam)

    # renaming columns for consistency
    df_na = df_na.rename(columns={'barcodes': 'ean', 'name': 'item_name'}).loc[:, ['ean', 'item_name']]
    df_latam = df_latam.rename(columns={'barcodes': 'ean', 'name': 'item_name'}).loc[:, ['ean', 'item_name']]
    
    return df_na, df_latam

def nlp_regex_cleaning(language_, data):
    print('NLP + Regex product name cleaning..')

    if language_ == 'en':
        stop_words = stopwords.words('english')
    elif language_ == 'es':
        stop_words = stopwords.words('spanish')

    regex_clean = r'(pm \d+\w+)|(pm \d+\.\d+)|(pm\d+\.\d+)|(\d+ pmp)|(pm\d+)|( \.+)|(pmp\d+.\d+)|(\d+pmp)|(pmp \d+)|(\d+.\d+ pm)'
    data_nlp = nlp_cleaning(data, stop_words, regex_clean)

    print(f'Percentage of unique products after NLP: {round(len(data_nlp.product_name.unique())/len(data_nlp.item_name.unique()), 3)}')

    return data_nlp.loc[:, ['ean', 'item_name', 'product_name']]

def read_canonical_file_to_match():
    print('Reading canonical file to match..')
    df_canonical = pd.read_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv')
    df_canonical['canonical_leader_lower'] = df_canonical['canonical_leader'].str.lower()
    df_match = df_canonical.loc[:, ['canonical_leader_lower']].rename(columns={'canonical_leader_lower': 'canonical_leader'})
    print(f'Initial canonical shape: {df_canonical.shape}')
    return df_canonical, df_match


def product_set_for_similarity(df, df_match):
    print('Concatenating items without barcode with items to match..')
    df_product_set = pd.DataFrame(data={'product_name': list(set(df['product_name'])) + list(set(df_match['canonical_leader']))})
    df_product_set = df_product_set.drop_duplicates().reset_index(drop=True)
    print(f'N° total products: {df_product_set.shape[0]}')
    return df_product_set

def cleaning_products_to_match(matches_df, df, df_match):
    print('Cleaning products to match..')
    products_to_match_list = list(set(df_match['canonical_leader']))
    products_with_ean = list(set(df['product_name']))
    matches_df = matches_df[(matches_df['product_name'].isin(products_to_match_list))&(matches_df['match'].isin(products_with_ean))].reset_index(drop=True)
    return matches_df

def one_match_per_product(df_clean):
    print('Determining the best match for each product..')
    df_one_match = df_clean.sort_values(['product_name', 'fuzz_ratio', 'package_ratio', 'similarity_score'], ascending=False).reset_index(drop=True)
    df_one_match = df_one_match.drop_duplicates('product_name').reset_index(drop=True)
    return df_one_match.loc[:, ['product_name', 'match']]

def add_barcodes_to_canonical(df, df_one_match, df_canonical):
    print('Adding barcodes to file..')

    # dictionary from raw ean --> item files to map barcodes
    ean_item_dict = dict(zip(df['product_name'], df['ean']))
    df_one_match['ean'] = df_one_match['match'].map(ean_item_dict)

    # adding barcodes into canonical set
    product_ean_dict = dict(zip(df_one_match['product_name'], df_one_match['ean']))
    df_canonical['ean'] = df_canonical['canonical_leader_lower'].map(product_ean_dict)
    df_canonical.drop('canonical_leader_lower', axis=1, inplace=True)
    
    print(f'% canonical items without barcode: {round((df_canonical[df_canonical["ean"].isna()].shape[0] / df_canonical.shape[0]) * 100, 2)}%')
    print(f'Final canonical shape: {df_canonical.shape}')

    return df_one_match, df_canonical


def main():
    
    # reading all sources of barcodes (depending on country)
    if language_ == 'en':
        df = read_uk_barcode_files()
        df_na, df_latam = read_cornershop_barcode_files()
        df = pd.concat([df, df_na], axis=0).drop_duplicates().reset_index(drop=True)
    else:
        df_na, df_latam = read_cornershop_barcode_files()
        df = df_latam.copy()

    # cleaning item names
    df = nlp_regex_cleaning(language_, df)

    print('Duplication')
    print(df[df.duplicated(['product_name'], keep=False)].sort_values('product_name'))

    # reading file to match
    if match_canonical:
        df_canonical, df_match = read_canonical_file_to_match()
        # concatenating products to single column to calculate similarities
        df_product_set = product_set_for_similarity(df, df_match)
        # # Appying TF-IDF method
        df_tf, tf_idf_matrix = tf_idf_method(df_product_set)
        # Applying cosine similarity to detect most similar products (potential group)
        matches_df = cosine_similarity_calculation(df_tf, tf_idf_matrix)
        # cleaning products to match
        clean_matches_df = cleaning_products_to_match(matches_df, df, df_match)
        # Calculating fuzzy ratios and keeping products with similarity above threshold_products
        df_similars = fuzzy_ratios(clean_matches_df, threshold_products=85)
        # calculating fuzzy ratios between product packages, keeping similarities above threshold_package
        df_clean = cleaning_by_package_similarity(df_similars, threshold_package=75, match_col='match', return_barcode=True)
        # rule: each product has one match (barcode)
        df_one_match = one_match_per_product(df_clean)
        # add know barcodes to canonical file
        df_one_match, df_canonical = add_barcodes_to_canonical(df, df_one_match, df_canonical)
        # saving canonical set (with barcodes)
        df_canonical.to_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv', index=False)
    


if __name__ == "__main__":
    main()