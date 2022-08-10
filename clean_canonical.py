
import pandas as pd


def classification_accuracy(df_clean):
    print('Calculating the number of correct group assignments..')
    df_true = df_clean[~df_clean['label'].isna()]
    df_true.label = df_true.label.str.strip().str.lower()
    print(f'Number of correct assignments: {round(df_true[df_true.label == "true"].shape[0]/len(df_true), 3)}')

def main():

    df = pd.read_csv('canonical_data/canonical_catalog.csv')
    df.columns = df.columns.str.strip().str.lower()

    # Number of correct assignments
    classification_accuracy(df)

    # select useful canonical columns
    df_canonical = df.loc[:, ['canonical_leader', 'brand', 'name', 'package', 'promotion']]
    df_canonical.insert(0, 'canonical_id', range(0, len(df_canonical)))

    df_canonical.to_csv('canonical_data/canonical_catalog.csv', index=False)


if __name__ == "__main__":
    main()