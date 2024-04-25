import re
import pandas as pd

req_fields = ["Patterns", "Method", "Priority"]

    
def check_p1(df: pd.DataFrame, config: dict, metric: str) -> pd.Series | None:
    rules = config[metric]
    if 'ColumnIndex' in rules:
        return df.iloc[:, rules['ColumnIndex']]
    namePattern = rules['ColumnNamePattern']
    
    options = check_rule(df, namePattern, type="name", p=1)
    if options is None or len(options.columns) == 0:
        return None
    
    if len(options.columns) > 1:
        raise ValueError(f'P1 Rule Multi-Match: rule: {namePattern}, matches: {list(options.columns)}')
        
    else:
        return options.iloc[:, 0]


    
def check_p2(df: pd.DataFrame, config: dict, metric: str) -> pd.DataFrame | None:
    rules = config[metric]
    namePattern = rules['ColumnNamePattern']
    
    options = check_rule(df, namePattern, type="name", p=2)
    if options is None or len(options.columns) == 0:
        return None
    return options
            


def check_rule(df: pd.DataFrame, pat: dict, type: str, p:int) -> pd.DataFrame | None:
    if pat is None:
        print(f'Warning: `pat` is None while checking {type} rule over {df.columns}')
        return df
    if df is None:
        print(f'Warning: `df` is None while checking {type} rule {pat}')
        return None
    if type.lower() not in ['name', 'value']:
        raise ValueError('Invalid Parameter Type: type should be either `name` or `value`')
    for rule in pat:
        options = __check_rule_name(df, rule, p=p) if type == 'name' else __check_rule_value(df, rule)
        if options is None:
            continue
        return options
    return None



def __check_rule_name(df: pd.DataFrame, rule: dict, p: int) -> pd.DataFrame | None:
    missing_fields = [field for field in req_fields if field not in rule]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    if rule['Priority'] != p:
        return None
    
    # 1. Set method and pattern
    isRegex = False if 'isRegex' not in rule else rule['isRegex']
    patterns = [p.lower() for p in rule['Patterns']]
    columns_lower = df.columns.str.lower()

    match rule['Method']:
        case 'Match':
            if isRegex:
                regex_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
                filtered_cols = [col for col in df.columns if any(re.fullmatch(regex, col) for regex in regex_patterns)]
            else:
                filtered_cols = df.columns[columns_lower.isin(patterns)]

        case 'Contain':
            if isRegex:
                regex_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
                filtered_cols = [col for col in df.columns if any(re.search(regex, col) for regex in regex_patterns)]
            else:
                filtered_cols = df.columns[columns_lower.str.contains('|'.join(patterns), case=False)]

        case _:
            raise ValueError(f"Invalid method: {rule['Method']}")

    # Look around neighbors
    if 'LookAround' in rule:
        key = rule['LookAround'].lower()
        if len(filtered_cols) == 1:
            return df[filtered_cols]
        elif len(filtered_cols) == 0:
            # 1. Generate the new pattern
            patterns_look = [pattern.replace(key, '').strip() for pattern in patterns if key in pattern]
            if not patterns_look:
                return None
            # 2. Find if there are any matches with the new pattern
            if rule['Method'] == 'Match':
                filtered_cols = df.columns[columns_lower.isin(patterns_look)]
            elif rule['Method'] == 'Contain':
                filtered_cols = df.columns[columns_lower.str.contains('|'.join(patterns_look), case=False)]
            else:
                return None
        # 3. Check whether neighbor contains such keyword
        # If there are multiple matches, can also use LookAround to further filter
        filtered_col_idxs = [df.columns.get_loc(col) for col in filtered_cols]
        res = []
        for i, col_id in enumerate(filtered_col_idxs):
            if col_id == 0 or col_id == len(df.columns) - 1:
                print(f'Warning: The LookAround is taken at the side of the table: \
                        col_id = {col_id}, col_name = {filtered_cols[i]}')
                continue
            if key in columns_lower[col_id - 1] or key in columns_lower[col_id + 1]:
                res.append(df.columns[col_id])
        return df[res]
    

    return df[filtered_cols]
        
def __check_rule_value(df: pd.DataFrame, rule: dict) -> pd.DataFrame | None:
    return df
