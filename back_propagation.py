import pandas as pd

# define parameters
country = 'uk'

def main():
    # reading file that has item_uuid and item_name match
    df_raw = pd.read_csv(f'back_propagation/{country}_uuid_name.csv')

    # reading file that has item_name and product_name (post NLP + regex cleaning)
    df_post_nlp = pd.read_csv(f'back_propagation/groups_{country}_back_propagation.csv')

    # reading file cleaned by agent task force
    df_clean = pd.read_csv('back_propagation/clean_booker_UK.csv')
    df_clean.columns = df_clean.columns.str.strip().str.lower()

    # DELETE
    print(f'% NAs: {round(df_clean[df_clean["label"].isna()].shape[0]/len(df_clean), 3)}')
    df_clean.loc[df_clean['label'].isna(), 'label'] = 'not_yet'
    df_clean['label'] = df_clean['label'].apply(lambda x: x.replace(' ', ''))

    # stat of correct matches
    print(f'Number of True matches: {round(df_clean[df_clean["label"] == "True"].shape[0]/len(df_clean), 4)}')
    
    # merging all
    df_map = df_raw.merge(df_post_nlp, how='inner', on='item_name')

    print(df_map)
    print(df_clean)
    
    #print(len(df_raw.item_name.unique())), print(len(df_post_nlp.item_name.unique())), print(len(df_post_nlp.product_name.unique()))


if __name__ == "__main__":
    main()