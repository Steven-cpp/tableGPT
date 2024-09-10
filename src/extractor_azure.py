import os
import pandas as pd
from error_code import *
from typing import Tuple, Optional
from pdf_preprocessor import update_map
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest



def to_pandas(result: AnalyzeResult, doc_map:pd.DataFrame, output_dir:str, proc_report_path:str):
    metadata_csv = {
        'report_path': [],
        'report_name': [],
        'table_type': [],
        'csv_path': [],
        'page_ori': []
    }

    # We can assume at most two tables in one page
    pages_with_table = []
    for table in result.tables:
        tableList = [['' for x in range(table.column_count)] for y in range(table.row_count)] 
        current_page_num = table.bounding_regions[0].page_number
        for cell in table.cells:
            if cell.row_index == 0 and "columnSpan" in cell:
                for offset in range(cell.column_span):
                    tableList[cell.row_index + 1][cell.column_index + offset] = cell.content.replace('\n', ' ') + ' '
            else:
                tableList[cell.row_index][cell.column_index] += cell.content.replace('\n', ' ')
        df = pd.DataFrame.from_records(tableList)
        # Set the header row
        df.columns = df.iloc[0] 
        df = df[1:]
        # Remove empty rows
        df = df[df.any(axis=1)]        
        # Write to excel only if dataframe has some data
        if not(df.empty):
            mask = (doc_map['processed_report_path'] == proc_report_path) & (doc_map['page_new'] == current_page_num)
            metadata = doc_map[mask]
            report_path = metadata['report_path'].iloc[0]
            report_name = metadata['report_name'].iloc[0]
            table_type = metadata['table_type'].iloc[0]
            page_ori = int(metadata['page_ori'].iloc[0])
            page_new = int(metadata['page_new'].iloc[0])
            fund_name = report_path.split('/')[-2]
            dir_path = os.path.join(output_dir, fund_name)
            if page_new not in pages_with_table:
                csv_path = os.path.join(dir_path, f'{page_new:02d}_{report_name}_{table_type}_{page_ori}.csv')
            else:
                csv_path = os.path.join(dir_path, f'{page_new:02d}_{report_name}_{table_type}_{page_ori}(1).csv')
            pages_with_table.append(page_new)
            update_map(metadata_csv,
                       report_path=report_path,
                       report_name=report_name,
                       table_type=table_type,
                       page_ori=page_ori,
                       csv_path=csv_path)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            df.to_csv(csv_path)

    return metadata_csv

def analyze_layout(report_path:str, doc_map:pd.DataFrame, output_dir='./output') -> Tuple[Optional[ErrorCode], Optional[dict]]:
    try:
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=os.getenv('ADI_ENDPOINT'), credential=AzureKeyCredential(os.getenv('ADI_KEY'))
        )
    except Exception as e:
        return CreateAzureServiceError(e), None

    try:
        with open(report_path, 'rb') as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=f.read())
            )
    except FileNotFoundError as e:
        return ReportNotFoundError(report_path), None
    except Exception as e:
        return AnalyzeDocumentError(e), None

    result: AnalyzeResult = poller.result()

    metadata = to_pandas(result, doc_map, output_dir, report_path)
    return None, metadata