import pandas as pd

def main():

    df_agents = pd.read_csv('agents_clean/agents_clean_uk_booker.csv')
    df_agents.columns = df_agents.columns.str.strip().str.lower()
    unique_members = list(set(df_agents[~df_agents["canonical_leader"].isna()]["member"]))   

    df_clean = df_agents[~df_agents['canonical_leader'].isna()]
    unique_leaders = list(set(df_clean["canonical_leader"]))

    # stats
    print(f'Number of members: {len(unique_members)}')
    print(f'Number of canonical_leaders:{len(unique_leaders)}')
    print(f'Percentage of leaders vs total: {round(len(unique_leaders)/len(unique_members), 3)}')

if __name__ == "__main__":
    main()