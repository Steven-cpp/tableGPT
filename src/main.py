import argparse
import logging
import json
import os
import pandas as pd
import numpy as np
from identifier import check_p1, check_p2

def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Make sure the first column is company name
    # Might require some efforts, but don't see any real cases for now

    # 2. Remove the table header, which is commonly in UPPER_CLASS
    pass

# Clean the extracted table to be consumed
def preprocess_identified(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.columns[1:]

    # 1. Convert percentage to floating number
    if 'Ownership' in df.columns:
        df['Ownership'] = df['Ownership'].str.rstrip('% ').astype(float) / 100

    # 2. Other Numeric values conversion
    for col in numeric_cols:
        df[col] = df[col].replace('[$,]', '', regex=True)\
                  .replace(r'\((.+?)\)', r'-\1', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Remove rows with all other non-numeric columns
    df.replace('', np.nan, inplace=True)
    df = df.dropna(subset=numeric_cols, how='all')

    # 4. Remove rows of empty company name or contain ['Total', 'Investment']
    df = df[~(df[df.columns[0]].str.contains('Total|Realized', case=False, na=False)\
            | (df.iloc[:, 0].isna()))]
    
    return df

def plot_summary(df: pd.DataFrame, gp: str, n: int) -> None:
    df['Accuracy'] = df['Sum'] / df['SumGT']
    cvg = sum(df['Src'].str.strip() == df['SrcGT'])
    print('=' * 23 + f' Extraction Summary Table of {gp} ' + '=' * 23)
    print(df)
    print('=' * 23 + f' Overall Coverage = {cvg / n}, END ' + '=' * 23)


def get_args():  
    parser = argparse.ArgumentParser(description='Consolidate financial tables into a master pandas DataFrame.')  
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV file.')  
    parser.add_argument('--config_path', type=str, required=True, help='Path to JSON configuration file.')  
    parser.add_argument('--test_cases_path', type=str, help='Path to JSON test cases file.')  
    return parser.parse_args()  


def extract_port(rule_path, test_path, csv_base_path='./res/final/'):
    rule_config = json.load(open(rule_path))
    case_config = json.load(open(test_path))

    for gp_type in case_config:
        fnames = [ i for i in os.listdir(csv_base_path) if gp_type in i and i.endswith('.csv')]
        if len(fnames) != 1:
            raise ValueError(f'Invalid gp_type: {gp_type}, cannot find it under {csv_base_path}')
        port, summary = __extract_port(csv_base_path + fnames[0], rule_config, case_config[gp_type])
        
        print()
        plot_summary(pd.DataFrame(summary), gp_type, len(case_config[gp_type].keys()))

        print()
        print('=' * 23 + f' Final Table of {gp_type} ' + '=' * 23)
        print(port)


def __extract_port(csv_path: str, rule_config: dict, case_config: dict) -> tuple:
    """
    Extract target metrics from GP report extracted tables specified by `case_config`.

    Args:
        csv_path (str): The path of the potential port summary table, extracted from GP report
        rule_config (dict): Identifier rules configuration.
        case_config (dict): Test case configuration.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - DataFrame 1: The extracted port summary.
            - DataFrame 2: The extraction summary table.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace('*', '', regex=False)

    res = pd.DataFrame()
    summary = {
        'Target': [],
        'Src': [],
        'Rule': [],
        'SrcGT': [],
        'Sum': [],
        'SumGT': []
    }
    
    for metric in rule_config:
        try:
            col = check_p1(df, rule_config, metric)
            if col is None:
                col = check_p2(df, rule_config, metric)
                if col is None:
                    continue
                summary['Rule'].append('P2')
                if len(col.columns) > 1:
                    print(f'WARNING: {metric} of {csv_path} multi-match via P2 rule: {col.columns}')
                    idx = input("Select the correct one:")
                    col = col.iloc[:, idx]
                else:
                    col = col.iloc[:, 0]
            else:
                summary['Rule'].append('P1')

            # Remove the columns after matching
            df.drop(columns=[col.name], inplace=True)

            # Append items into summary table
            summary['Target'].append(metric)
            summary['Src'].append(col.name)
            srcGT = case_config[metric]['Source'] if metric in case_config else np.nan
            sumGT = case_config[metric]['Sum'] if metric in case_config else np.nan
            summary['SrcGT'].append(srcGT)
            summary['SumGT'].append(sumGT)
            col.name = metric
            res = pd.concat([res, col], axis=1)

        except ValueError as e:
            logging.warning('Multi-Match: %s: %s', metric, e)
    
    # If no columns were matched, return an empty DataFrame
    if res.empty:
        return summary
    
    port = preprocess_identified(res)
    sumArr = [np.nan] + list(port.sum(numeric_only=True))
    summary['Sum'] = sumArr

    return port, summary
    

if __name__ == "__main__":
    logging.info('Extracting Portfolio Summary from GP Reports')
    rule_path = 'config.json'
    test_path = './test/case.json'
    extract_port(rule_path, test_path)