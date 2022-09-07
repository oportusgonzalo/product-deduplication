
import pandas as pd


country = ''

def main():

    df = pd.read_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv')

    print(f'N° initial canonical IDs: {len(list(set(df["canonical_id"])))}')
    print(f'N° initial canonical leaders: {len(list(set(df["canonical_leader"])))}')
    
    # to keep first canonical leaders that have been verified - remove duplicated data
    df = df.drop_duplicates(subset=['canonical_leader']).reset_index(drop=True)

    print(f'N° final canonical IDs: {len(list(set(df["canonical_id"])))}')
    print(f'N° final canonical leaders: {len(list(set(df["canonical_leader"])))}')
    
    df.to_csv(f'canonical_data/{country}/{country}_canonical_catalog.csv', index=False)


if __name__ == "__main__":
    main()