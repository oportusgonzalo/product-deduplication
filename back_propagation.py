import pandas as pd

# define parameters
country = 'uk'
parent_chain = 'nisa'
threshold_products = 85
threshold_package = 75
exists_canonical_catalog = True

def main():
    if exists_canonical_catalog:
        # reading file that has item_name and product_name (post NLP + regex cleaning)
        df_back = pd.read_csv(f'back_propagation/raw_vs_clean_{country}_{parent_chain}_products_{threshold_products}_{threshold_package}.csv')

        # reading file with direct matches
        df_direct = pd.read_csv(f'bivariate_outputs/direct_matches_{country}_{parent_chain}_{threshold_products}_{threshold_package}.csv')
        print(df_direct)

        # agents clean groups
        df_clean = pd.read_csv(f'agents_clean/agents_clean_{country}_booker.csv') # add parameter parent_chain
        df_clean.columns = df_clean.columns.str.strip().str.lower()

        # Number of correct assignments
        df_true = df_clean[~df_clean['label'].isna()]
        df_true.label = df_true.label.str.strip().str.lower()
        print(f'Number of correct assignments: {round(df_true[df_true.label == "true"].shape[0]/len(df_true), 3)}')

        # useful columns
        df_clean = df_clean.loc[: ,['member', 'canonical_leader', 'brand', 'name', 'package', 'promotion']]

        # merging datasets
        df_merge = df_back.merge(df_clean, how='inner', left_on='product_name', right_on='member')
        df_merge.drop(['product_name', 'member'], axis=1, inplace=True)


    print(df_back)
    print(df_clean)
    print(df_merge)
    pass

    
if __name__ == "__main__":
    main()