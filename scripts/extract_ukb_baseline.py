"""
This script extracts all the baseline fields (instance 0) 
from the master UKB Showcase tsv file and saves these to a feather file.
This process is not perfect as instance 0 is sometimes used for non-baseline data (e.g. health outcomes).
This is rectified in later processing by removing irrelevant categories.
Note: has high memory requirement and runs for a long time.
"""

import pandas as pd
import yaml

# Load YAML config file
with open("../config.yml", "r") as file:
    config = yaml.safe_load(file)

def select_baseline_columns():
    """
    Selects all the baseline columns from the UKB master tsv file.
    """
    print("Selecting baseline columns...")

    all_cols = pd.read_csv(config['paths']['ukb_showcase_raw_dataset'], sep="\t", nrows=0, low_memory=False)
    fields_metadata = pd.read_csv("field.txt", sep="\t")

    # Select only columns at instance 0 (i.e. baseline)
    all_baseline_cols = [col for col in all_cols.columns.tolist()[1:] if int(col.split('.')[2]) == 0]

    # Remove fields with 'compound' data type
    compound_fields = [int(i) for i in fields_metadata.loc[fields_metadata['value_type'] == 101, 'field_id'].values]
    all_baseline_cols = [col for col in all_baseline_cols if int(col.split('.')[1]) not in compound_fields]
    all_baseline_cols = ['f.eid'] + all_baseline_cols

    # Define data types based on UKB documentation
    dtype_dict = {
        11: 'Int16',
        21: 'category',
        22: 'category',
        31: 'Float32',
        41: 'string',
        51: 'datetime64[ns]',
        61: 'datetime64[ns]',
    }

    # Set appropriate pandas data type for each column
    col_dtypes = {'eid': 'Int32'}
    datetime_cols = []

    for col in all_baseline_cols[1:]:
        col_dtype = dtype_dict[int(fields_metadata.loc[fields_metadata['field_id'] == int(col.split('.')[1]), 'value_type'].values[0])]
        if col_dtype == 'datetime64[ns]':
            datetime_cols.append(col)
        else:
            col_dtypes[col] = col_dtype
    
    return all_baseline_cols, col_dtypes, datetime_cols


def main():
    print("Starting UKB baseline data extraction...")
    
    all_baseline_cols, col_dtypes, datetime_cols = select_baseline_columns()
    
    print("Loading baseline data...")
    baseline_data = pd.read_csv(config['paths']['ukb_showcase_raw_dataset'], sep="\t", usecols=all_baseline_cols, dtype=col_dtypes, parse_dates=datetime_cols)
    baseline_data.columns = [col[2:] for col in baseline_data.columns]

    print("Saving to feather...")
    baseline_data.to_feather('all_showcase_baseline.feather')

    print("Done!")

if __name__ == "__main__":
    main()