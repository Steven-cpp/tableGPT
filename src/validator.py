import pandas as pd

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