
import pandas as pd

from static import *

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


pd.set_option('display.max_columns', 6)

language_='en'
match_canonical = True


def read_barcode_files():
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
    df_canonical = pd.read_csv('canonical_data/uk/uk_canonical_catalog.csv')
    df_canonical['canonical_leader_lower'] = df_canonical['canonical_leader'].str.lower()
    df_match = df_canonical.loc[:, ['canonical_leader_lower']].rename(columns={'canonical_leader_lower': 'canonical_leader'})
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
    ean_item_dict = dict(zip(df['product_name'], df['ean']))
    df_one_match['ean'] = df_one_match['product_name'].map(ean_item_dict)

    print(ean_item_dict)
    print(df_one_match)
    print(df_one_match[df_one_match['ean'].isna()])

    print(df)
    print(df_canonical)

    pass


def main():
    # reading all sources of barcodes
    df = read_barcode_files()
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
        add_barcodes_to_canonical(df, df_one_match, df_canonical)
        


if __name__ == "__main__":
    main()