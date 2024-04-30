import argparse
import logging
import json
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from extractor import extract_pdf
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
    
    # 5. Value self-validation
    cross_check_cols = ['UnrealizeValue', 'RealizedValue', 'Total']
    if sum(df.columns.isin(cross_check_cols)) == len(cross_check_cols):
        df['TotalExp'] = df['UnrealizedValue'] + df['RealizedValue']
        if np.all(df['TotalExp'] == df['Total']):
            logging.info('Table Validation Passed !!!')
        else:
            logging.warning('Table Validation Failed >_<')

    return df

def plot_summary(df: pd.DataFrame, gp: str, n: int) -> None:
    df['Accuracy'] = df['Sum'] / df['SumGT']
    cvg = sum(df['Src'].str.strip() == df['SrcGT'])
    logging.info('Extraction Summary Table of %s:\n %s', gp, df)
    print('=' * 23 + f' Extraction Summary Table of {gp} ' + '=' * 23)
    print(df)
    print('=' * 23 + f' Overall Coverage = {cvg / n}, END ' + '=' * 23)


def get_args():  
    parser = argparse.ArgumentParser(description='Consolidate financial tables into a master pandas DataFrame.')  
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV file.')  
    parser.add_argument('--config_path', type=str, required=True, help='Path to JSON configuration file.')  
    parser.add_argument('--test_cases_path', type=str, help='Path to JSON test cases file.')  
    return parser.parse_args()  


def extract_port(rule_path, test_path, base_path='./res/final/'):
    """_summary_

    Args:
        rule_path (_type_): _description_
        test_path (_type_): _description_
        base_path (str, optional): _description_. Defaults to './res/final/'.

    Raises:
        ValueError: _description_
    """
    rule_config = json.load(open(rule_path))
    case_config = json.load(open(test_path))
    # ports, period = []

    for gp_type in case_config:
        _case = case_config[gp_type]
        gp_dir = [name for name in os.listdir(base_path) \
                if os.path.isdir(os.path.join(base_path, name)) and gp_type in name]
        if len(gp_dir) != 1:
            logging.warning(f'Invalid GP Type: {gp_type}, multiple matches or cannot find the folder under {base_path}')
            continue
        
        gp_dir = gp_dir[0]
        fnames = [ i for i in os.listdir(os.path.join(base_path, gp_dir)) if i.endswith('.csv')]

        for fname in fnames:
            port, summary = __extract_port(os.path.join(base_path, gp_dir, fname), rule_config, _case)
            if port.empty or len(port.columns) < 3:
                logging.info(f'Ignore invalid table {fname}.')
                continue

            if summary.empty:
                logging.warning(f'Failed to extract metrics from {fnames[0]}')
                continue

            # Append unmatched metrics at the end
            gt_targets = set(_case.keys())
            fp = gt_targets.difference(set(summary['Target']))
            for i in fp:
                fp_row = pd.Series({'SrcGT': _case[i]['Source'], 
                                    'SumGT': _case[i]['Sum'],
                                    'Rule': _case[i]['Rule'],
                                    'Target': i})
                fp_row = fp_row.reindex(summary.columns, fill_value=np.nan)
                summary = pd.concat([summary, fp_row.to_frame().T], ignore_index=True)

            # Show the extraction summary table
            print()
            plot_summary(summary, fname, len(gt_targets))

            # Show the extracted portfolio table
            logging.info('Final Table of %s: \n %s', gp_type, port)


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
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        logging.warning(f'Failed to read {csv_path}: {e}')
        return pd.DataFrame(), pd.DataFrame()
    
    if len(df) < 4:
        return pd.DataFrame(), pd.DataFrame()
    
    df.columns = df.columns.str.replace('*', '', regex=False)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    pat_series = 'Series [A-H]-?[0-9]?(?![a-zI-Z])'
    is_layered = df.apply(lambda x: x.str.contains(pat_series, regex=True).any()\
                          if x.dtype == 'O' else False, axis=0).any()
    if is_layered:
        logging.warning(f'Unsupported Report Type: {csv_path} may be splitted by series.')
        return pd.DataFrame(), pd.DataFrame()

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
                    print(f'{metric} of {csv_path} multi-match via P2 rule: {col.columns}')
                    idx = int(input("Select the correct one:"))
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
        return pd.DataFrame(), pd.DataFrame()
    
    port = preprocess_identified(res)
    sumArr = [np.nan] + list(port.sum(numeric_only=True))
    summary['Sum'] = sumArr

    return port, pd.DataFrame(summary)
  

if __name__ == "__main__":
    logging.basicConfig(filename='output.log', \
                        level=logging.INFO, \
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    pdf_dir = './docs'
    csv_dir = './output'
    rule_path = 'config.json'
    test_path = './test/case.json'

    logging.info('1. Extracting Tables from PDF File')
    file_num = extract_pdf(pdf_dir, csv_dir)
    logging.info('2. Identifying Portfolio Summary Table')

    logging.info('3. Processing the Extracted Table')

    extract_port(rule_path, test_path)