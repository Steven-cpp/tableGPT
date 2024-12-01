import logging
import json
import re
import os
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
    df['company_name'] = df['company_name'].fillna(value='')
    df['security_type'] = df['security_type'].fillna(value='')
    while row_id < len(df) - 1:
        company_name = ''
        start = row_id
        while df.loc[row_id + 1, 'security_type'] != '' and len(df.loc[row_id + 1, 'company_name']) < 2:
            company_name += df.loc[row_id, 'company_name']
            row_id += 1
            if row_id >= len(df) - 1:
                break
        company_name += ' ' + df.loc[row_id, 'company_name']

        df.loc[start:row_id, 'company_name'] = company_name.strip()
        row_id += 1
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
                continue
                # raise UnsupportedC2ReportTypeWarning("No parent detected for the security type.")
            row_dict = row.to_dict()
            if row[[COMPANY_NAME_COL]].isna().iloc[0]:
                parent = ''
                row_dict[COMPANY_NAME_COL] = ''
            if 'total' in row_dict[COMPANY_NAME_COL].lower():
                res_dicts.append(row_dict)
                continue
            row_dict[SECURITY_TYPE_COL] = row_dict[COMPANY_NAME_COL]
            row_dict[COMPANY_NAME_COL] = parent
            res_dicts.append(row_dict)
    
    return pd.DataFrame(data=res_dicts)


def __preprocess_total_sit(df: pd.DataFrame) -> pd.DataFrame:
    """Identify company-level total line in the SIT, fill `company_name` and set `is_total`
    Therefore, the subtotal line can be easily idenfitied by `is_total=True` and `row_id` < MAX(row_id)

    Basis:
        - The company-level total line is right above the next company
        with both `company_name` and `security_type` empty
        - The company-level total line is the top line of the company (signaled by Null `security_type`
        and Not-Null `ownership`)

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    row_id = 0
    LEN_COMPANY_NAME_MIN = 4
    while row_id < len(df) - 2:
        company_name = ''
        while not pd.isna(df.loc[row_id, 'security_type']) and df.loc[row_id, 'security_type'] != '':
            company_name = df.loc[row_id, 'company_name']
            row_id += 1
            # We stop at the last 3rd row, to ensure we do not check the last row
            if row_id >= len(df) - 3:
                break
        next_company_name = df.loc[row_id + 1, 'company_name']
        if df.loc[row_id, 'company_name'] == '' and len(next_company_name) > LEN_COMPANY_NAME_MIN and 'total' not in next_company_name.lower():
            df.loc[row_id, 'is_total'] = True
            df.loc[row_id, 'company_name'] = company_name
        row_id += 1
    
    if 'ownership' not in df.columns:
        return df
    
    df.replace('', np.nan, inplace=True)
    mask_null_security = df['security_type'].isna()
    mask_not_null_ownership = df['ownership'].notna()
    mask_subtotal = mask_null_security & mask_not_null_ownership
    df.loc[mask_subtotal, 'is_total'] = True

    return df

def __preprocess_total(df: pd.DataFrame) -> pd.DataFrame:
    """Identify total line in the table, and set `isTotal` column

    Args:
        df (pd.DataFrame): Preprocessed extracted table

    Returns:
        pd.DataFrame: the original df with a new `isTotal` column
    """
    rule_contain = ['Total Investments', 'Total Portfolio Investments', 'Total Portfolio', 'Total Unrealized +', 'Total Fund']
    rule_equal = ['Total', '', 'Totals']
    # Check last two rows only
    rule_lastNRows = 2

    def is_total(header) -> bool:
        if pd.isna(header):
            return True
        if not isinstance(header, str):
            return False
        header = header.lower().strip()
        is_contain = any(s.lower() in header for s in rule_contain)
        is_equal = any(s.lower() == header for s in rule_equal)
        return is_contain or is_equal

    for i in range(rule_lastNRows):
        idx = - (i + 1)
        if idx < -len(df):
             break
        if is_total(df.iloc[idx][COMPANY_NAME_COL]):
            df.at[df.index[idx], 'is_total'] = True
            break

    return df


# Clean the extracted table to be consumed
def preprocess_identified(df: pd.DataFrame) -> pd.DataFrame:
    pat_series = 'Series [A-Z](?:-\d+)?|Class [A-Z]|Common Stock|Preferred'
    is_layered = df.apply(lambda x: x.str.contains(pat_series, regex=True, case=False).any()\
                          if x.dtype == 'O' else False, axis=0).any()
    if is_layered or SECURITY_TYPE_COL in df.columns:
        if SECURITY_TYPE_COL not in df.columns:
            # Extract hidden security type as a separate column first
            df = extract_hidden_security_type(df)
        else:
            # Fill empty company name
            df = extend_company_name(df)
    
    df['is_total'] = False
    if SECURITY_TYPE_COL in df.columns:
        df = __preprocess_total_sit(df)

    numeric_cols = df.columns.drop([COMPANY_NAME_COL, SECURITY_TYPE_COL, INVESTMENT_DATE], errors="ignore")

    def transform_ownership(s):
        if isinstance(s, str) and '%' in s:
            s = s.replace('%', '')
            s = re.sub(r'[\[\]·]', '', s)
            s = re.sub(r'\((.+?)\)', r'-\1', s)
            try:
                return float(s) / 100
            except:
                return np.nan
        return np.nan

    # 1. Convert percentage to floating number
    if 'ownership' in df.columns:
        df['ownership'] = df['ownership'].apply(transform_ownership)
    
    if 'gross_irr' in df.columns:
        df['gross_irr'] = df['gross_irr'].apply(transform_ownership)

    # 2. Other Numeric values conversion
    for col in numeric_cols:
        df[col] = df[col].replace(r'[$,x]', '', regex=True)\
                  .replace(r'\((.+?)\)', r'-\1', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Remove rows with all other non-numeric columns
    df.replace('', np.nan, inplace=True)
    df = df.dropna(subset=numeric_cols, how='all')
    if len(df) == 0 or df['company_name'].dtype != 'O':
        return pd.DataFrame()

    # 4. Identify total line at the bottom
    df = __preprocess_total(df)

    # 5. Remove rows of empty company name or contain ['Total', 'Investment']
    name_mask = df[df.columns[0]].str.contains('Total|Realized', case=False, na=False) | (df.iloc[:, 0].isna())
    real_total_mask = df['is_total']
    df = df[~name_mask | real_total_mask]
    
    # 6. Value self-validation
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

def __contain_pst_keywords(text: str, rule_config, n=2) -> bool:
    """ Check whether the string contains keywords that may construct PST

    Args:
        text (str): the texts to check
        rule_config (json): the rule defined by `config.json`
        n (int, optional): the least number of keywords to be matched. Defaults to 4.

    Returns:
        bool: whether the page contains sufficient keywords
            * True: there are sufficient keywords, should further check
            * False: keywords not sufficient, ignore this page
    """
    target_metrics = ['total_cost', 'unrealized_value', 'realized_value', 'total', 'gross_moic']
    mask = [False] * len(rule_config)
    text_lower = text.lower()

    for idx, metric in enumerate(rule_config):
        if metric not in target_metrics:
            continue
        rule = rule_config[metric]
        if 'ColumnNamePattern' not in rule:
            continue
        namePatterns = rule['ColumnNamePattern']
        for rule in namePatterns:
            if 'isRegex' in rule:
                continue
            patterns = rule['Patterns']
            for pat in patterns:
                if pat.lower() in text_lower:
                    mask[idx] = True
                    text_lower = text_lower.replace(pat.lower(), '')
                    continue
    return True if sum(mask) >= n else False


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
        # All the missing values are treated as `np.nan`
        df = pd.read_csv(csv_path, index_col=False)
    except Exception as e:
        logging.warning(f'Failed to read {csv_path}: {e}')
        raise InvalidCSVError()
    
    if len(df) < 4:
        raise InvalidTableWarning()
    
    # Azure will create an index column for all the csvs
    if PDF_EXTRACTION_API == 'Azure':
        df = df.drop(columns=df.columns[0])
    
    ## 0. Check if the header is invalid
    invalid_cols = [col for col in df.columns if not __is_valid_column_name(col)]
    if len(invalid_cols) > 0:
        raise InvalidTableWarning()
    mask_empty_cols = df.columns.str.contains('Unnamed')
    df.columns = [col if not mask_empty_cols[i] else '' for i, col in enumerate(df.columns)]
    if sum(mask_empty_cols) >= len(mask_empty_cols):
        df.columns = df.iloc[0, :].fillna('')
        df = df.iloc[1:, :].reset_index(drop=True)

    ## 1. Column Name Cleaning
    # If the column row is mis-identified, then the label
    subheader = ' '.join([s for s in df.iloc[0, :].to_list() if isinstance(s, str)])
    is_misplaced = __contain_pst_keywords(subheader, rule_config, n=2)
    if is_misplaced:
        df.columns = [df.columns[i] + ' ' + df.iloc[0, i] if type(df.iloc[0, i]) is str else df.columns[i] for i in range(len(df.columns))]
        df.columns = df.columns.str.strip()
        df = df.drop([0]).reset_index(drop=True)
    
    # Make sure there are no duplicate column names
    if sum(df.columns.duplicated()) > 1:
        raise InvalidTableWarning()

    # Clear `:unselected:` or `:selected:` from the cell
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.replace(':unselected:', '').str.replace(':selected:', '')
        except Exception as e:
            continue
    
    df.columns = df.columns.str.replace('*', '', regex=False)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    # Left strip irrelevant columns
    # 1. Not SIT Table
    pat_series = 'Series [A-Z](?:-\d+)?|Class [A-Z]|Common Stock|Preferred Stock'
    is_layered = df.apply(lambda x: x.str.contains(pat_series, regex=True, case=False).any()\
                          if x.dtype == 'O' else False, axis=0).any()
    strip_columns_n = 2
    for i in range(strip_columns_n):
        # 2. AND many missing values in the first column
        cnt_nan = df.iloc[:, 0].isna()
        is_mostly_empty = sum(cnt_nan) / len(cnt_nan) > 0.5 and sum(cnt_nan) > 3
        is_company_col = 'company' in df.columns[0].lower()
        is_invalid_col = sum(pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()) > 2
        # 3. Do the strip
        if is_invalid_col or (not is_layered and is_mostly_empty and not is_company_col):
            df = df.iloc[:, 1:]
            continue
        break

    if len(df.columns) < 3:
        return pd.DataFrame(), pd.DataFrame()
    
    ## 2. Totals row cleaning
    # If the totals are split into a separate column,
    # try to append this column into the left column.
    mask_interpolated = ~df.iloc[:, 0].isna() & ~df.iloc[:, 1].isna()
    if sum(mask_interpolated) == 0 and not is_company_col:
        # Merge strings in 1st column and 2nd column
        mask_to_merge = ~df.iloc[:, 1].isna()
        if mask_to_merge.sum() > 0:
            df.loc[mask_to_merge, df.columns[0]] = df.loc[mask_to_merge, df.columns[1]]
        # Now drop the second column since it's merged into the first
        df.drop(df.columns[1], axis=1, inplace=True)


    # 3. Fill empty column names with `Unnamed: {idx}`
    df.columns = [col if isinstance(col, str) and col != '' else 'Unnamed: ' + str(idx) for idx, col in enumerate(df.columns)]

    res = pd.DataFrame()
    metric_dict = {
        'target': [],
        'src': [],
        'rule': [],
    }

    # Ignore summary table
    if 'summary of' in df.columns[0].lower():
        return pd.DataFrame(), pd.DataFrame()

    for metric in rule_config:
        try:
            col = check_p1(df, rule_config, metric)
            if col is None:
                col = check_p2(df, rule_config, metric)
                if col is None:
                    continue
                metric_dict['rule'].append('P2')
                if len(col.columns) > 1:
                    raise P2RuleMultiMatchWarning(metric)
                else:
                    col = col.iloc[:, 0]
            else:
                metric_dict['rule'].append('P1')

            # Remove the columns after matching
            df.drop(columns=[col.name], inplace=True)

            # Append items into summary table
            metric_dict['target'].append(metric)
            metric_dict['src'].append(col.name)
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


def __is_valid_column_name(col_name) -> bool:
    if col_name is None:
        return True
    col_name = str(col_name)
    if col_name.isdigit():
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(filename='output.log', \
                        level=logging.INFO, \
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    rule_path = 'config.json'
    report_name = 'Sequoia Capital U.S. Venture 2010 - 06.2023'
    report_path = './docs/' + report_name + '.pdf'
    csv_dir = './output'

    report_csv_dir = csv_dir + '/' + report_name

    load_dotenv('.env')

    logging.info('1. Extracting Tables from PDF File')

    report_paths = [
        './docs/Source Code Growth Fund I L.P. - Q1 2022 - QR .pdf',
        # './docs/TA XIV-B Q3 2023 Report.pdf'
    ]

    test_csv_paths = [
        './output/docs/01_Source Code Growth Fund I L.P. - Q1 2022 - QR .pdf_SIT_1.csv'
    ]

    # processed_report_path, metadata = process_docs(report_paths, rule_path)
    # metadata = pd.DataFrame(metadata)
    # metadata['processed_report_path'] = processed_report_path
    # err, csv_records = analyze_layout(processed_report_path, metadata)
    # if err: 
    #     print(err)
    #     raise RuntimeError()
    # csv_records = pd.DataFrame(csv_records)
    # logging.info('Done: Tables are extracted from PDF files')
    # logging.info(csv_records)

    # logging.info('2. Identifying Portfolio Summary Table')

    # logging.info('3. Processing the Extracted Table')

    # for csv_path in csv_records['csv_path']:
    for csv_path in test_csv_paths:
        csv_fn = csv_path.split('\\')[-1]
        error, port, metric_summary = extract_port(rule_path, csv_path)
        if error is None:
            print(csv_fn)
            print(metric_summary)
        else:
            print(f"{csv_fn}: {error}")
