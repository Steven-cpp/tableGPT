import pandas as pd
import re
import logging
import pandas as pd
from cleanco import basename
from typing import Optional
from thefuzz import process, fuzz
from util import connect_sql

logger = logging.getLogger(__name__)

def to_camel_case(s):
    words = s.split()
    camel_case_words = [word.capitalize() for word in words]
    camel_case_string = ' '.join(camel_case_words)
    return camel_case_string

def clean_portco_name(name: str) -> str:
    # match dba/fka name in a case insensitive way
    pattern_dba = re.compile(r'(?:dba|aka)\s+([^\)]+)', re.IGNORECASE)
    match = pattern_dba.search(name)
    if match:
        name = match.group(1).strip()
    
    # remove brackets
    pattern_comma = r',.*'
    pattern_bracket = r'\((.*)'
    pattern_end_digit = r' \d+$'
    pattern_noise = r'^\* '
    name = re.sub(pattern_comma, '', name)
    name = re.sub(pattern_bracket, '', name)
    name = re.sub(pattern_end_digit, '', name)
    name = re.sub(pattern_noise, '', name)

    # If all in upper case, to camel case
    if name == name.upper():
        name = to_camel_case(name)
    
    return basename(name).replace('"', '')

def match_inso_name(portco: pd.DataFrame, t=85) -> Optional[int]:
    """
    Match extracted companies with INSO database by name

    Args:
        portco (pd.DataFrame): Portfolio companies extracted from GP reports
        t (int): Threshold for fuzzy matching

    Returns:
        int: Company ID if matched, None otherwise

    Note:
        The function matches the portfolio company name by taking the *front part* as an identifier, which is
        defined as the substring in lower case without the trailing word, e.g:
            - FRONT_PART('Neural') = 'neural'
            - FRONT_PART('Neural Science') = 'neural'
            - FRONT_PART('Neural Science Inc.') = 'neural science'
        However, if the front part is shorter than 4 characters, then it should take the entire string as the identifier. 
        It first matches by identifier: if the identifier equals to the INSO company identifier, then it is a match,
        and we add the INSO company full-name into the list of potential matches for each extracted entry.
        In the end, we use fuzzy matching to compare the extracted company name with the list of potential matches:
            - If there is only one match, we consider it as a match.
            - If there are multiple matches, we use fuzzy matching to score each candidate and select the one with the highest score:
                * If there are multiple highest scores, we consider it as a failed match.
                * If the score is lower than `t`, we consider it as a failed match.
              Finally, if the score is higher than `t`, we consider it as a successful match.
    """
    MIN_LEN_IDENTIFIER = 4

    try:
        portco_inso = spark.sql("""SELECT CompanyID AS company_id, CompanyName AS company_name_inso
                                    FROM GP_Reports.last_company_enriched """).toPandas()
    except Exception:
        conn = connect_sql(env='prd')
        portco_inso = pd.read_sql_query("""SELECT CompanyID AS company_id, CompanyName AS company_name_inso
                                           FROM Powerapps.LastCompanyEnriched """, conn)
        pass

    portco_inso['company_name_inso'] = portco_inso['company_name_inso'].apply(clean_portco_name)
    portco_ext = portco[['company_name']].drop_duplicates()

    # 1. Extract identifier from company name
    portco_ext['identifier'] = portco_ext['company_name'].apply(lambda x: ' '.join(x.split()[:-1]).lower() if len(x.split()) > 1 and len(''.join(x.split()[:-1])) >= MIN_LEN_IDENTIFIER else x.lower())
    portco_inso['identifier'] = portco_inso['company_name_inso'].apply(lambda x: ' '.join(x.split()[:-1]).lower() if len(x.split()) > 1 and len(''.join(x.split()[:-1])) >= MIN_LEN_IDENTIFIER else x.lower())
    
    # 2. Match by identifier
    portco_ext = portco_ext.merge(portco_inso, how='left', on='identifier')
    portco_ext = portco_ext.drop(columns=['identifier'])

    # 3. Aggregate potential matches for each extracted company
    portco_ext = portco_ext.groupby('company_name').agg({'company_name_inso': list}).reset_index()

    # 4. Fuzzy matching
    portco_ext['candidates'] = portco_ext.apply(lambda x: process.extract(x['company_name'], x['company_name_inso'], scorer=fuzz.ratio), axis=1)
    portco_ext['candidate'] = portco_ext['candidates'].apply(lambda x: x[0][0] if (len(x) > 1 and x[0][1] > t and x[0][1] > x[1][1])
                                                                or len(x) == 1 else None)
    portco_ext['company_id'] = portco_ext['candidate'].apply(lambda x: portco_inso.loc[portco_inso['company_name_inso'] == x, 'company_id'].values[0] if x else None)
    portco_ext = portco_ext.drop(columns=['candidate'])

    # 5. Merge with original extracted companies
    portco = portco.merge(portco_ext, how='left', on='company_name')

    return portco


def match_inso(mode: str, csv_path=None) -> pd.DataFrame:
    logger.info("Step 7: Matching Portfolio Company with INSO")

    if mode == 'full':
        port_validated_table = 'portfolio_company_validated'
    elif mode == 'partial':
        port_validated_table = 'int_portfolio_company_validated'
    else:
        raise ValueError('Invalid Parameter: `mode` should be either `full` or `partial')
    
    if csv_path is None:
        port_company_validated = spark.sql("""SELECT * FROM GP_Reports.{port_validated_table}""").toPandas()
        port_company_validated['company_id'] = port_company_validated['company_id'].astype('Int64')
    else:
        port_company_validated = pd.read_csv(csv_path)

    port_company_validated = port_company_validated.dropna()
    # Filter out rows with company_name as floating point values using regex
    pattern_out = r'^[^A-Za-z]*$'
    port_company_validated = port_company_validated[~port_company_validated['company_name'].str.match(pattern_out)]
    port_company_validated.loc[:, 'company_name'] = port_company_validated['company_name'].apply(clean_portco_name)

    port_name_matched = match_inso_name(port_company_validated)
    port_company_matched = port_company_validated.merge(port_name_matched[['company_name', 'company_id']], how='left', on='company_name')

    if csv_path is None:
        port_company_matched_spark = spark.createDataFrame(data=port_company_matched, \
                                                        schema=port_company_matched.columns.to_list())
        port_company_matched_spark.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable(f'{port_validated_table}')

    mask_matched = set(list(port_company_matched['company_id']))
    mask_all = set(list(port_company_matched['company_name']))
    logger.info("Success: %d out of %d companies get matched", len(mask_matched), len(mask_all))
    return port_company_matched


def validate_metrics(df: pd.DataFrame):
    """
    Validate metrics or fill missing metrics based on column relationship

    Args
        df (pd.DataFrame): portfolio summary table

    """
    total_val = ['total', 'realized_value', 'unrealized_value']
    moic_val = ['total', 'total_cost', 'gross_moic']

    # Initialize validation columns
    df['is_validated'] = None
    filled_metrics = pd.DataFrame()

    # 1. Validate `Total`
    mask_val = df[total_val].notna().all(axis=1)

    # Set 'is_validated' for rows where all values are present
    df.loc[mask_val, 'is_validated'] = df.loc[mask_val, 'total'] == (df.loc[mask_val, 'realized_value'] 
                                                                     + df.loc[mask_val, 'unrealized_value'])

    # Fill missing values for total
    mask_fill = df[total_val].isna().sum(axis=1) == 1
    for col in total_val:
        mask = df[mask_fill][total_val].isna()
        filled_metrics_new = df.loc[mask_fill & mask[col], ['csv_path']]
        if col == 'total':
            # Locate rows where ONLY `total` is missing
            df.loc[mask_fill & mask[col], 'total'] = df.loc[mask_fill & mask[col], 'realized_value'] \
                                                     + df.loc[mask_fill & mask[col], 'unrealized_value']
            filled_metrics_new['target'] = 'total'

        elif col == 'realized_value':
            df.loc[mask_fill & mask[col], 'realized_value'] = df.loc[mask_fill & mask[col], 'total'] \
                                                              - df.loc[mask_fill & mask[col], 'unrealized_value']
            filled_metrics_new['target'] = 'realized_value'
        elif col == 'unrealized_value':
            df.loc[mask_fill & mask[col], 'unrealized_value'] = df.loc[mask_fill & mask[col], 'total'] \
                                                                - df.loc[mask_fill & mask[col], 'realized_value']
            filled_metrics_new['target'] = 'unrealized_value'

        if filled_metrics.empty:
            filled_metrics = filled_metrics_new
        else:
            filled_metrics = pd.concat([filled_metrics, filled_metrics_new], axis=0)    

    # 2. Validate `Gross MOIC`
    mask_val_moic = df[moic_val].notna().all(axis=1)
    mask_fill_moic = df[moic_val].isna().sum(axis=1) == 1

    # Set 'is_validated' for rows where all values are present for MOIC
    df.loc[mask_val_moic, 'is_validated'] = df.loc[mask_val_moic, 'is_validated'] & (
        df.loc[mask_val_moic, 'gross_moic'] == df.loc[mask_val_moic, 'total'] / df.loc[mask_val_moic, 'total_cost']
    )

    # Fill missing values for MOIC
    for col in moic_val:
        mask = df[mask_fill_moic][moic_val].isna()
        filled_metrics_new = df.loc[mask_fill_moic & mask[col], ['csv_path']]
        if col == 'total':
            df.loc[mask_fill_moic & mask[col], 'total'] = df.loc[mask_fill_moic & mask[col], 'gross_moic'] \
                                                          * df.loc[mask_fill_moic & mask[col], 'total_cost']
            filled_metrics_new['target'] = 'total'
        elif col == 'total_cost':
            df.loc[mask_fill_moic & mask[col], 'total_cost'] = df.loc[mask_fill_moic & mask[col], 'total'] \
                                                               / df.loc[mask_fill_moic & mask[col], 'gross_moic']
            filled_metrics_new['target'] = 'total_cost'
        elif col == 'gross_moic':
            df.loc[mask_fill_moic & mask[col], 'gross_moic'] = df.loc[mask_fill_moic & mask[col], 'total'] \
                                                               / df.loc[mask_fill_moic & mask[col], 'total_cost']
            filled_metrics_new['target'] = 'gross_moic'

        if filled_metrics.empty:
            filled_metrics = filled_metrics_new
        else:
            filled_metrics = pd.concat([filled_metrics, filled_metrics_new], axis=0) 
    
    filled_metrics['rule'] = 'VA'

    return df, filled_metrics
    