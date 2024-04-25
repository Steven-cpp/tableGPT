import argparse
import logging
import json
import re
import pandas as pd
import numpy as np
from identifier import check_p1


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.columns[1:]

    # 1. Numeric values conversion
    for col in numeric_cols:
        df[col] = df[col].replace('[$,]', '', regex=True)\
                  .replace(r'\((.+?)\)', r'-\1', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Remove rows with all other non-numeric columns
    df.replace('', np.nan, inplace=True)
    df = df.dropna(subset=numeric_cols, how='all')

    # 3. Remove rows of empty company name or contain ['Total', 'Investment']
    df = df[~df[df.columns[0]].str.contains('Total|Investment', case=False, na=False)]
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

def main():
    csv_name = 'Sequoia PS_0'
    path_csv = f'./res/final/{csv_name}.csv'
    gp = re.sub('[ ]?[\(.*\)]?_\d', '', csv_name)

    rule_config = json.load(open('config.json'))
    case_config = json.load(open('./test/case.json'))

    truth = case_config[gp]
    df = pd.read_csv(path_csv)

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
        # summary['SrcGT'].append()
        # summary['SumGT'].append()
        try:
            col = check_p1(df, rule_config, metric)
            if col is not None:
                summary['Rule'].append('P1')
                summary['Target'].append(metric)
                summary['Src'].append(col.name)
                srcGT = truth[metric]['Source'] if metric in truth else np.nan
                sumGT = truth[metric]['Sum'] if metric in truth else np.nan
                summary['SrcGT'].append(srcGT)
                summary['SumGT'].append(sumGT)
                col.name = metric
                res = pd.concat([res, col], axis=1)

        except ValueError as e:
            logging.warning('Multi-Match: %s: %s', metric, e)
    
    # If no columns were matched, return an empty DataFrame
    if res.empty:
        return summary
    
    port = preprocess(res)
    sumArr = [np.nan] + list(port.sum(numeric_only=True))
    summary['Sum'] = sumArr

    print()
    plot_summary(pd.DataFrame(summary), gp, len(truth.keys()))

    print()
    print('=' * 23 + f' Final Table of {gp} ' + '=' * 23)
    print(port)
    

if __name__ == "__main__":
    logging.info('Extracting Portfolio Summary from GP Reports')
    main()