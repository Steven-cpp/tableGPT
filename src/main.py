import logging
import pandas as pd
from dotenv import load_dotenv
from table_processor import extract_port
from pdf_preprocessor import process_docs
from extractor_azure import analyze_layout

# Ideally, the notebook should directly call the function defined here
# And get the desired pandas dataframe as output

if __name__ == "__main__":
    logging.basicConfig(filename='output.log', \
                        level=logging.INFO, \
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    rule_path = 'config.json'
    report_name = 'Sequoia Capital U.S. Venture 2010 - 06.2023'
    report_path = './docs/' + report_name + '.pdf'
    csv_dir = './output'

    report_csv_dir = csv_dir + '/' + report_name

    load_dotenv('.env')

    logger.info('1. Extracting Tables from PDF File')

    report_paths = [
        './docs/Venrock Associates VI - Q2 2022 - Quarterly Report.pdf',
        # './docs/TA XIV-B Q3 2023 Report.pdf'
    ]

    test_csv_paths = [
        './output/docs/01_Sequoia Capital India Growth Fund I - Q2 2022 - FS.pdf_SIT_5.csv'
    ]

    processed_report_path, metadata = process_docs(report_paths, rule_path)
    metadata = pd.DataFrame(metadata)
    metadata['processed_report_path'] = processed_report_path
    err, csv_records = analyze_layout(processed_report_path, metadata)
    if err:
        print(err)
        raise RuntimeError()
    csv_records = pd.DataFrame(csv_records)
    logger.info('Done: Tables are extracted from PDF files')
    logger.info(csv_records)

    logger.info('2. Identifying Portfolio Summary Table')

    logger.info('3. Processing the Extracted Table')

    for csv_path in csv_records['csv_path']:
    # for csv_path in test_csv_paths:
        csv_fn = csv_path.split('\\')[-1]
        error, port, metric_summary = extract_port(rule_path, csv_path)
        if error is None:
            print(csv_fn)
            print(metric_summary)
        else:
            print(f"{csv_fn}: {error}")
    
