"""
Contains useful processing functions used in jupyter notebooks.
"""

import pandas as pd
import numpy as np

def display_included_cats(included_cols: list) -> tuple[dict, pd.DataFrame]:
    """
    Converts UKB's categorical hierarchy into a tree & pretty pandas dataframe.
    Input: Takes a list of columns as input (in 0.0.0 [id, instance, array] format) and marks included primary categories with a tick.
    Output: Returns both a tree (as dict in the format {parent: [children]}) and a pretty pandas dataframe.
    """
    # Metadata files from UKB Schema - https://biobank.ndph.ox.ac.uk/showcase/schema.cgi
    cat_hierarchy = pd.read_csv("../data/ukb_metadata/catbrowse.txt", sep="\t") # Schema 13
    cat_metadata = pd.read_csv("../data/ukb_metadata/category.txt", sep="\t") # Schema 3
    fields_metadata = pd.read_csv("../data/ukb_metadata/field.txt", sep="\t") # Schema 1
    
    # Build a dictionary to represent the tree
    tree = {}
    for _, row in cat_hierarchy.iterrows():    
        parent, child = int(row["parent_id"]), int(row["child_id"])
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(child)
    
    # Recursively get all paths
    def get_paths(node, path):
        if node not in tree:
            return [path]  # Leaf node. End recursion and return path.
        paths = []
        for child in tree[node]:
            paths.extend(get_paths(child, path + [child]))
        return paths
    
    # Find the root nodes (parents without a parent)
    roots = set(cat_hierarchy["parent_id"]) - set(cat_hierarchy["child_id"])
    paths = []
    for root in roots:
        if root in tree:  # Only process valid roots
            paths.extend(get_paths(root, [root]))
    
    # Convert paths into a DataFrame
    max_depth = max(map(len, paths))  # Find the longest path
    hierarchy_df = pd.DataFrame([p + [''] * (max_depth - len(p)) for p in paths]) # Pad shorter paths with empty strings
    
    # Rename columns to reflect hierarchy levels
    hierarchy_df.columns = [f"Level {i+1}" for i in range(max_depth)]
    
    # Replace category IDs with titles, and add tick if included category
    all_categories = pd.concat([cat_hierarchy['parent_id'], cat_hierarchy['child_id']], ignore_index=True).drop_duplicates()
    included_fields = list(set([int(col.split('.')[0]) for col in included_cols]))
    included_categories = fields_metadata.loc[fields_metadata['field_id'].isin(included_fields), 'main_category'].unique().tolist()

    category_mapping = {}
    for cat_id in all_categories:
        try:
            # Convert to: Title (ID) format
            title = cat_metadata.loc[cat_metadata['category_id'] == cat_id, 'title'].values[0] + f" ({cat_id})"
        except:
            title = f'Unknown category ({cat_id})'
        # Add tick if included category
        if cat_id in included_categories:
            title = title + ' ✅'
        category_mapping[cat_id] = title
    
    hierarchy_df = hierarchy_df.replace(category_mapping)
    # Set index to all columns to make it look pretty
    hierarchy_df = hierarchy_df.set_index([f"Level {i+1}" for i in range(max_depth)])
    return tree, hierarchy_df



def get_downstream_cats(parent_cat: int, tree: dict) -> list:
    """ Get all categories downstream of a parent category """
    downstream_cats = []
    if parent_cat in tree:
        for child in tree[parent_cat]:
            downstream_cats.append(child)
            [downstream_cats.append(cat) for cat in get_downstream_cats(child, tree)]
    return downstream_cats

def get_cat_and_downstream(parent_cat: int, cat_tree: dict) -> list:
    """ Add all downstream categories to parent category """
    return [parent_cat] + get_downstream_cats(parent_cat, cat_tree)


def remove_cat_cols(columns: list, categories: list) -> list:
    """ Removes all columns associated with a list of categories """
    fields_metadata = pd.read_csv("../data/ukb_metadata/field.txt", sep="\t") # UKB Schema 1
    fields_to_remove = fields_metadata.loc[fields_metadata['main_category'].isin(categories), 'field_id'].tolist()
    new_columns = remove_fields(columns, fields_to_remove)
    return new_columns

def remove_fields(columns: list, fields_to_remove: list) -> list:
    """ Removes all columns associated with a list of fields """
    new_columns = [col for col in columns if int(col.split('.')[0]) not in fields_to_remove]
    n_cols_removed = len([col for col in columns if int(col.split('.')[0]) in fields_to_remove])
    print(f"Removed {n_cols_removed} columns")
    return new_columns

def display_arrayed_fields(columns: list) -> pd.DataFrame:
    """
    For a list of UKB columns, identify those that consist of arrayed fields. 
    Present these as a DataFrame grouped by primary category, with basic details for each field and an array count. 
    """

    # Encodings of field data types
    dtype_dict = {
        11: 'Integer',
        21: 'Single choice',
        22: 'Multi choice',
        31: 'Decimal',
        41: 'String',
        51: 'Datetime',
        61: 'Datetime',
        101: 'Compound',
        201: 'Blob'
    }

    fields_metadata = pd.read_csv("../data/ukb_metadata/field.txt", sep="\t") # Schema 1
    cat_metadata = pd.read_csv("../data/ukb_metadata/category.txt", sep="\t") # Schema 3
    arrayed_fields = {}
    
    for col in columns:
        col_parts = col.split('.')
        field_id = int(col_parts[0])
        array_n = int(col_parts[2])
        if array_n > 0:
            field = fields_metadata[fields_metadata['field_id'] == field_id]
            # Add to dict using field title (and ID) as keys and [category name (ID), data type, array count] as values
            cat_details = cat_metadata[cat_metadata['category_id'] == field['main_category'].values[0]]
            arrayed_fields[field_id] = [field['title'].values[0], dtype_dict[field['value_type'].values[0]], f"{cat_details['title'].values[0]} ({cat_details['category_id'].values[0]})", int(col_parts[2]) + 1]
    
    # Convert to df
    arrayed_fields = pd.DataFrame([[k, v[0], v[1], v[2], v[3]] for k, v in arrayed_fields.items()], columns=['Field ID', 'Field title', 'Field type', 'Primary category', 'Number of arrays'])
    arrayed_fields = arrayed_fields.set_index(['Primary category',  'Field title']).sort_values(by=['Primary category', 'Field ID'])
    return arrayed_fields