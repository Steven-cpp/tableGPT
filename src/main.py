import logging
import json
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv 
from typing import Tuple, Optional
from pdf_preprocessor import process_docs
from extractor_azure import analyze_layout
from identifier import check_p1, check_p2
from error_code import *

INVESTMENT_DATE = 'investment_date'
SECURITY_TYPE_COL = 'security_type'
COMPANY_NAME_COL = 'company_name'
PDF_EXTRACTION_API = 'Azure'


def extend_company_name(df: pd.DataFrame) -> pd.DataFrame:
    """Fill empty company name if security type already exists

    Args:
        df (pd.DataFrame): the potential SIT of type C1

    Returns:
        pd.DataFrame: the final SIT with filled company name
    """
    row_id = 0
    while row_id < len(df):
        company_name = ''
        start = row_id
        while df.iloc[row_id, :]['security_type']:
            row = df.iloc[row_id, :]
            company_name += row['company_name'] if row['company_name'] else ''
            row_id += 1
            if row_id >= len(df):
                break
        regex_dba = re.compile(r'\(dba (.+?)\)')
        mat = regex_dba.search(company_name)
        if mat:
            company_name = mat.group(1)
        row_id += 1
        df.iloc[start: row_id, 'company_name'] = company_name.strip()
    return df

def extract_hidden_security_type(df: pd.DataFrame) -> pd.DataFrame:
    """Extracy hidden security type from typical C2 schedule of investment table(SIT).
       After extraction, the C2 table would turn to type C1.

    Args:
        df (pd.DataFrame): the potential SIT of type C2

    Raise:
        UnsupportedC2ReportTypeWarning: the potential SIT schema is not supported
 
    Returns:
        pd.DataFrame: the same SIT of type C1
    """
    parent = ''
    res_dicts = []
    for _, row in df.iterrows():
        if row.notnull().sum() == 1:
            # If the first column is null, then there is an error
            if not row.notnull().iloc[0]:
                continue
            parent = row.iloc[0]
        else:
            if parent == '':
                raise UnsupportedC2ReportTypeWarning("No parent detected for the security type.")
            row_dict = row.to_dict()
            if row[[COMPANY_NAME_COL]].isna().iloc[0] or 'total' in row_dict[COMPANY_NAME_COL].lower():
                continue
            row_dict[SECURITY_TYPE_COL] = row_dict[COMPANY_NAME_COL]
            row_dict[COMPANY_NAME_COL] = parent
            res_dicts.append(row_dict)
    
    return pd.DataFrame(data=res_dicts)


# Clean the extracted table to be consumed
def preprocess_identified(df: pd.DataFrame) -> pd.DataFrame:
    pat_series = 'Series [A-Z](?:-\d+)?|Class [A-Z]|Common Stock|Preferred Stock'
    is_layered = df.apply(lambda x: x.str.contains(pat_series, regex=True).any()\
                          if x.dtype == 'O' else False, axis=0).any()
    if is_layered:
        if SECURITY_TYPE_COL not in df.columns:
            # 0. Extract hidden security type as a separate column first
            df = extract_hidden_security_type(df)
        else:
            # 0. Fill empty company name
            df = extend_company_name(df)

    numeric_cols = df.columns.drop([COMPANY_NAME_COL, SECURITY_TYPE_COL, INVESTMENT_DATE], errors="ignore")

    # 1. Convert percentage to floating number
    if 'ownership' in df.columns:
        df = df[df['ownership'].str.contains('%', na=True)]
        df['ownership'] = df['ownership'].str.replace(r'[\[\]]', '', regex=True)\
                          .str.rstrip('% ').astype(float) / 100

    # 2. Other Numeric values conversion
    for col in numeric_cols:
        df[col] = df[col].replace('[$,]', '', regex=True)\
                  .replace(r'\((.+?)\)', r'-\1', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Remove rows with all other non-numeric columns
    df.replace('', np.nan, inplace=True)
    df = df.dropna(subset=numeric_cols, how='all')
    if len(df) == 0 or df['company_name'].dtype != 'O':
        return pd.DataFrame()

    # 4. Remove rows of empty company name or contain ['Total', 'Investment']
    df = df[~(df[df.columns[0]].str.contains('Total|Realized', case=False, na=False)\
            | (df.iloc[:, 0].isna()))]
    
    # 5. Value self-validation
    cross_check_cols = ['unrealize_value', 'realized_value', 'total']
    if sum(df.columns.isin(cross_check_cols)) == len(cross_check_cols):
        df['total_exp'] = df['unrealize_value'] + df['realized_value']
        if np.all(df['total_exp'] == df['total']):
            logging.info('Table Validation Passed !!!')
        else:
            logging.warning('Table Validation Failed >_<')

    return df


def extract_port(rule_path: str, csv_path: str) -> Tuple[Optional[ErrorCode], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """ Identify and extract portfolio summary table from the given csv file

    Args:
        rule_path (str): rule to identify the column name
        csv_path (str): the csv file to be identified        

    Return:
        err (ErrorCode): errors occurred in extracting the report   
        port (pd.DataFrame): extracted portfolio table
        metric (pd.DataFrame): details of extracted metrics
    """
    report_name = csv_path.split('/')[-2]
    rule_config = json.load(open(rule_path))

    try:
        port, metric = __extract_port(csv_path, rule_config)
    except ErrorCode as e:
        return e, None, None
    
    if port.empty or len(port.columns) < 3:
        logging.info(f'Ignore invalid table {report_name}.')
        return InvalidTableWarning(), None, None

    # Show the extracted portfolio table
    logging.info('Final Table of %s: \n %s', csv_path, port)
    return None, port, metric


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
        df = pd.read_csv(csv_path, index_col=False)
    except Exception as e:
        logging.warning(f'Failed to read {csv_path}: {e}')
        raise InvalidCSVError()
    
    if len(df) < 4:
        raise InvalidTableWarning()
    
    # Azure will create an index column for all the csvs
    if PDF_EXTRACTION_API == 'Azure':
        df = df.drop(columns=df.columns[0])

    # If the real column is in second row, set the second row as the new header
    unnamed_mask = df.columns.str.contains('Unnamed:')
    if sum(unnamed_mask) >= df.shape[1] // 2:
        # Set the second row as the new header
        df.columns = df.iloc[0]
        # Drop the first two rows (misplaced header and actual header row)
        df = df.drop([0]).reset_index(drop=True)
    
    df.columns = df.columns.str.replace('*', '', regex=False)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
    # Ignore empty column names
    df = df.loc[:, ~df.columns.isna()]

    res = pd.DataFrame()
    metric_dict = {
        'Target': [],
        'Src': [],
        'Rule': [],
    }

    for metric in rule_config:
        try:
            col = check_p1(df, rule_config, metric)
            if col is None:
                col = check_p2(df, rule_config, metric)
                if col is None:
                    continue
                metric_dict['Rule'].append('P2')
                if len(col.columns) > 1:
                    raise P2RuleMultiMatchWarning(metric)
                else:
                    col = col.iloc[:, 0]
            else:
                metric_dict['Rule'].append('P1')

            # Remove the columns after matching
            df.drop(columns=[col.name], inplace=True)

            # Append items into summary table
            metric_dict['Target'].append(metric)
            metric_dict['Src'].append(col.name)
            col.name = metric
            res = pd.concat([res, col], axis=1)

        except ValueError as e:
            raise P1RuleMultiMatchError(metric)
    
    # If no columns were matched, return an empty DataFrame
    if res.empty:
        return pd.DataFrame(), pd.DataFrame()

    if len(res.columns) < 3:
        raise InvalidTableWarning()

    port = preprocess_identified(res)
    if port.empty:
        return pd.DataFrame(), pd.DataFrame()

    return port, pd.DataFrame(metric_dict)
  

if __name__ == "__main__":
    logging.basicConfig(filename='output.log', \
                        level=logging.INFO, \
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    rule_path = 'config.json'
    report_name = 'Lightspeed India Partners III Q2 2023 - Quarterly report-nowm'
    report_path = './docs/' + report_name + '.pdf'
    csv_dir = './output'

    report_csv_dir = csv_dir + '/' + report_name

    load_dotenv('.env')

    logging.info('1. Extracting Tables from PDF File')

    report_paths = [
        './docs/08_Insight Venture Partners VII - Q3 2023 - QR - PST.pdf',
        # './docs/TA XIV-B Q3 2023 Report.pdf'
    ]

    test_csv_paths = [
        './output/Lightspeed India Partners III Q2 2023 - Quarterly report-nowm/table_PST_20.csv'
    ]

    processed_report_path, metadata = process_docs(report_paths)
    metadata = pd.DataFrame(metadata)
    err, csv_records = analyze_layout(processed_report_path, metadata)
    if err:
        raise RuntimeError()
    csv_records = pd.DataFrame(csv_records)
    logging.info('Done: Tables are extracted from PDF files')
    logging.info(csv_records)

    # logging.info('2. Identifying Portfolio Summary Table')

    # logging.info('3. Processing the Extracted Table')

    for csv_path in csv_records['csv_path']:
    # for csv_path in test_csv_paths:
        csv_fn = csv_path.split('\\')[-1]
        error, port, metric_summary = extract_port(rule_path, csv_path)
        if error is None:
            print(csv_fn)
            print(metric_summary)
        else:
            print(f"{csv_fn}: {error}")
