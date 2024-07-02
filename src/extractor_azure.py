import os
import pandas as pd
from error_code import *
from typing import Tuple, Optional
from pdf_preprocessor import update_map
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest



def to_pandas(result: AnalyzeResult, doc_map:pd.DataFrame, output_dir:str):
    metadata_csv = {
        'report_path': [],
        'report_name': [],
        'table_type': [],
        'csv_path': [],
        'page_ori': []
    }
    for table in result.tables:
        tableList = [[None for x in range(table.column_count)] for y in range(table.row_count)] 
        current_page_num = table.bounding_regions[0].page_number
        merged_row_ids = []
        for cell in table.cells:
            # 1. if rowSpan > 1: row_index += 1 for rows == rowIndex
            if "row_span" in cell:
                if cell.row_span == 2:
                    merged_row_ids.append(cell.row_index)
                elif cell.row_span > 2:
                    raise UnsupportedMergedCellError(cell.content)

            # 2. AND if columnSpan > 1: duplicate to separate cells downwards
            if cell.row_index in merged_row_ids:
                cell.row_index += 1
            if "column_span" in cell:
                for offset in range(cell.column_span):
                    tableList[cell.row_index][cell.column_index + offset] = cell.content.replace('\n', ' ')
            else:
                tableList[cell.row_index][cell.column_index] = cell.content.replace('\n', ' ')

        df = pd.DataFrame.from_records(tableList)
        # Set the header row
        df.columns = df.iloc[0] 
        df = df[1:]
        # Remove empty rows
        df = df[df.any(axis=1)]        
        # Write to excel only if dataframe has some data
        if not(df.empty):
            metadata = doc_map[doc_map['page_new'] == current_page_num]
            report_path = metadata['report_path'].iloc[0]
            report_name = metadata['report_name'].iloc[0]
            table_type = metadata['table_type'].iloc[0]
            page_ori = int(metadata['page_ori'].iloc[0])
            dir_path = os.path.join(output_dir, report_name.replace('.pdf', ''))
            csv_path = os.path.join(dir_path, f'table_{table_type}_{page_ori}.csv')
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

    metadata = to_pandas(result, doc_map, output_dir)
    return None, metadata