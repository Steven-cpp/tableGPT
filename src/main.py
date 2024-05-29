import argparse
import logging
import json
import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from extractor import extract_pdf
from identifier import check_p1, check_p2
from error_code import *


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


def update_GT(test_case_path: str) -> Optional[ErrorCode]:
    """ Update GT in `source.matched_metric` given predefined test case
    If there already exists the record with the same `(report_path, csv_path, target)`, 
    we just need to update the GT column. Otherwise, we need to insert a new record.

    Args:
        test_case_path (str): path of predefined test case

    Returns:
        ErrorCode: if None, then no error occurred; else error occurred in the process
            
    """
    # 1. Read from `test_case_path`, get (report_path, csv_path, source_gt, sum_gt, target)
    # 2. Do the database update


def extract_port(rule_path: str, csv_path: str) -> Tuple[Optional[ErrorCode], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """ Identify and extract portfolio summary table from the given csv file

    Args:
        rule_path (str): rule to identify the column name
        csv_path (str): the csv file to be identified

    Raises:
        ValueError: _description_
    """
    report_name = csv_path.split('/')[-2]
    rule_config = json.load(open(rule_path))

    try:
        port, summary = __extract_port(csv_path, rule_config)
    except ErrorCode as e:
        return e, None, None
    
    if port.empty or len(port.columns) < 3:
        logging.info(f'Ignore invalid table {report_name}.')
        return InvalidTableWarning(), None, None

    # Show the extracted portfolio table
    logging.info('Final Table of %s: \n %s', report_name, port)
    return None, port, summary


def __extract_port(csv_path: str, rule_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract target metrics from GP report extracted tables specified by `case_config`.

    Args:
        csv_path (str): The path of the potential port summary table, extracted from GP report
        rule_config (dict): Identifier rules configuration.

    Raises:
        - InvalidCSVError()
        - InvalidTableWarning()
        - UnsupportedReportTypeWarning()
        
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - port: The extracted port summary.
            - summary: The extracted metric summary table.
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        logging.warning(f'Failed to read {csv_path}: {e}')
        raise InvalidCSVError()
    
    if len(df) < 4:
        raise InvalidTableWarning()
    
    df.columns = df.columns.str.replace('*', '', regex=False)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    pat_series = 'Series [A-H]-?[0-9]?(?![a-zI-Z])'
    is_layered = df.apply(lambda x: x.str.contains(pat_series, regex=True).any()\
                          if x.dtype == 'O' else False, axis=0).any()
    if is_layered:
        logging.warning(f'Unsupported Report Type: {csv_path} may be splitted by series.')
        raise UnsupportedReportTypeWarning()

    res = pd.DataFrame()
    summary = {
        'Target': [],
        'Src': [],
        'Rule': [],
        'Sum': [],
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
                    raise P2RuleMultiMatchWarning()
                else:
                    col = col.iloc[:, 0]
            else:
                summary['Rule'].append('P1')

            # Remove the columns after matching
            df.drop(columns=[col.name], inplace=True)

            # Append items into summary table
            summary['Target'].append(metric)
            summary['Src'].append(col.name)
            col.name = metric
            res = pd.concat([res, col], axis=1)

        except ValueError as e:
            raise P1RuleMultiMatchError()
    
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
    
    report_name = 'SEQUOIA CAPITAL INDIA V - Q2 2023 - FS.pdf'
    report_path = './docs/' + report_name
    csv_dir = './output'
    rule_path = 'config.json'

    logging.info('1. Extracting Tables from PDF File')
    error, records = extract_pdf(pdf_path, csv_dir)
    if error is None:
        records = pd.DataFrame(records)
        print(records)
    else:
        print(error)

    logging.info('2. Identifying Portfolio Summary Table')

    logging.info('3. Processing the Extracted Table')

    report_name = 'Battery Ventures'
    report_csv_dir = csv_dir + '/' + report_name
    csv_fns = os.listdir(report_csv_dir)

    for csv_fn in csv_fns:
        error, port, metric_summary = extract_port(rule_path, report_csv_dir + '/' + csv_fn)
        if error is None:
            print(metric_summary)
        else:
            print(error)
