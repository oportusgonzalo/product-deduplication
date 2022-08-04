import pandas as pd
pd.set_option('display.max_columns', 15)

# define parameters
country = 'uk'
parent_chain = 'booker'
threshold_products = 85
threshold_package = 75

def main():
    # reading file that has item_uuid and item_name match (source/raw file)
    df_raw = pd.read_csv(f'data/{country}_{parent_chain}_uuid_name.csv')

    # reading file that has item_name and product_name (post NLP + regex cleaning)
    df_post_nlp = pd.read_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')

    # reading file cleaned by agent task force
    df_clean = pd.read_csv('agents_clean/clean_booker_UK.csv')
    df_clean.columns = df_clean.columns.str.strip().str.lower()

    # DELETE
    #print(f'% NAs: {round(df_clean[df_clean["label"].isna()].shape[0]/len(df_clean), 3)}')
    df_clean.loc[df_clean['label'].isna(), 'label'] = 'not_yet'
    df_clean['label'] = df_clean['label'].apply(lambda x: x.replace(' ', ''))

    # stat of correct matches
    print(f'Number of True matches: {round(df_clean[df_clean["label"] == "True"].shape[0]/len(df_clean), 4)}')

    
if __name__ == "__main__":
    main()