# ## 6. Validation

# ### 6.1 Products added
print(f'Number of groups: {len(groups_df["group_id"].unique())}')

original_products = df_tf['product_name'].unique()
len(original_products)

added_products = pd.unique(groups_df[['leader', 'member']].values.ravel('K'))
len(added_products)

not_added = []
for prod_ in original_products:
    if prod_ not in added_products:
        not_added.append(prod_)

print(f'Number of products without group: {len(not_added)}')


# ### Who are they?
not_added[-10:]


# ### 6.2 Duplicated leaders / members ?? (in 2 or more groups)

# uniques: group_id - leader
leaders_df = groups_df[['group_id', 'leader']].drop_duplicates().reset_index(drop=True)
# duplicated leaders
leaders_df[leaders_df['leader'].duplicated() == True]

# uniques: group_id - member
members_df = groups_df[['group_id', 'member']].drop_duplicates().reset_index(drop=True)
# duplicated members
members_df[members_df['member'].duplicated() == True]


# ### 6.3 Adding not matched products

# Products not added to the groups dataframe are because previously they demonstrated low similarity on the clusters generated with TF-IDF + Cosine Similarity layer. This why they are added as "individual groups".
max_id = groups_df['group_id'].max()
not_added_df = pd.DataFrame(data={
                    'group_id': range(max_id, max_id + len(not_added)),
                    'leader': not_added,
                    'member': not_added})

# concat to groups_df
groups_df = pd.concat([groups_df, not_added_df], axis=0).reset_index(drop=True)

# concat to track df
track_df = pd.concat([track_df, not_added_df.loc[:, ['group_id', 'member']]], axis=0).reset_index(drop=True)


len(groups_df['leader'].unique()), len(groups_df['member'].unique())

groups_df[(groups_df['leader'].str.contains('coca'))|(groups_df['member'].str.contains('coca'))][:60]


# ### 6.6 Are all leaders in members?

groups_df.head()

leaders_list = list(set(groups_df.leader))
members_list = list(set(groups_df.member))

len(leaders_list), len(members_list)

len(list(set(~groups_df[groups_df['member'].isin(leaders_list)]['member'])))

not_member = []
for leader_ in leaders_list:
    if leader_ not in members_list:
        not_member.append(leader_)

len(not_member)