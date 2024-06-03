from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException

import json
import os
import re
import pandas as pd
import time
import zipfile
import logging
import shutil
from dotenv import load_dotenv
from typing import Tuple, Optional
from error_code import *

root_base = '/lakehouse/default/Files/End-to-end Demo'


def handle_exception(exception_type, exception_message, status_code):
    logging.info(exception_type)
    if status_code is not None:
        logging.info(status_code)
    logging.info(exception_message)

# Extract tables from specified pdf
def extract_table(pdf_path: str, target_dir: str, cont: ExecutionContext) -> ErrorCode:
    try:
        start = time.time()
        fn = pdf_path.split('/')[-1]
        logging.info('Extracting tables from %s', pdf_path.split('/')[-1])
        #Set operation input from a source file.
        source = FileRef.create_from_local_file(pdf_path)
        extract_pdf_operation = ExtractPDFOperation.create_new()
        extract_pdf_operation.set_input(source)

        #Build ExtractPDF options and set them into the operation
        extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
            .with_element_to_extract(ExtractElementType.TABLES)\
            .with_table_structure_format(TableStructureType.CSV) \
            .build()

        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
        result: FileRef = extract_pdf_operation.execute(cont)
        zip_path = f"{target_dir}/{fn[:-4]}.zip"
        # Save the result to the specified location.
        result.save_as(zip_path)
        end = time.time()
        
        print('Table extraction of %s finishes: %d sec.', fn, int(end - start))
        return zip_path

    except (ServiceApiException, ServiceUsageException, SdkException) as service_api_exception:
        raise AdobeAPIRequestError(service_api_exception.message,
                                   getattr(service_api_exception, 'status_code', None))
    
    
# Unzip the package obtained via API and rename it
def unzip(zip_path: str, csv_target_dir: str, cache=False) -> Tuple[list, int]:
    zipFn = zip_path.split('/')[-1]
    dest_dir = os.path.join(csv_target_dir, f'{zipFn[:-4]}')

    if not cache:
        if not zipFn.endswith('zip'):
            raise UnzipError(f'Invalid zip path {zip_path}')
        
        # 1. Extract zip file to the current working dir
        ext_dir = zip_path[:-4]
        with zipfile.ZipFile(zip_path, 'r') as zipFile:
            zipFile.extractall(ext_dir)
        os.remove(zip_path)

        # 2. Move Tables to target directory
        if not os.path.exists(os.path.join(ext_dir, 'tables')):
            raise NoTableExtractedWarning()

        if not os.path.exists(dest_dir):
            try:
                os.mkdir(dest_dir)
            except OSError as e:
                logging.error(f'Failed to create dir {dest_dir}: {e}')
                raise UnzipError(f'{e}')
            
        for tab in os.listdir(os.path.join(ext_dir, 'tables')):
            fileId = int(re.search(r'\d+', tab).group())
            shutil.copy(os.path.join(ext_dir, 'tables', tab), \
                        os.path.join(dest_dir, f'{zipFn[:-4]}_{fileId}.csv'))
        
        # 3. Save `structuredData`` as reference
        shutil.copy(os.path.join(ext_dir, 'structuredData.json'),
                os.path.join(dest_dir, 'structuredData.json'))
    
    # 4. Read `structuredData.json` to generate record metadata
    records = []
    with open(os.path.join(dest_dir, 'structuredData.json')) as f:
        metadata = json.load(f)
    page_count = metadata['extended_metadata']['page_count']
    elements = [element for element in metadata['elements'] if 'filePaths' in element]
    for idx, e in enumerate(elements):
        record = {
            'csv_path': os.path.join(dest_dir, f'{zipFn[:-4]}_{idx}.csv'),
            'col_num': e['attributes']['NumCol'],
            'row_num': e['attributes']['NumRow'],
            'page': e['Page']
        }
        records.append(record)

    
    return records, page_count


def extract_pdf(pdf_path: str, csv_dir: str) -> Tuple[Optional[ErrorCode], Optional[list], Optional[int]]:
    """Extract tables from specified pdf file, then save each csv table
       to the specified folder. If the target csv/fn[:-4] folder already exists,
       return the result set directly.

    Args:
        pdf_path (str): The pdf file to be processed
        csv_dir (str): The folder to store all extracted csv tables

    Returns:
        ErrorCode: Error code from pdf extraction API call
        dict: Extracted csv record
    """
    load_dotenv('{root_base}/.env')
    # TODO Change `target_dir` to '/tmp'
    target_dir = './tmp'

    # Initial setup, create credentials instance.
    credentials = Credentials.service_principal_credentials_builder()\
        .with_client_id(os.getenv('PDF_SERVICES_CLIENT_ID'))\
        .with_client_secret(os.getenv('PDF_SERVICES_CLIENT_SECRET')).build()

    # Create an ExecutionContext using credentials and create a new operation instance.
    execution_context = ExecutionContext.create(credentials)

    report_csv_dir = csv_dir + '/' + pdf_path.split('/')[-1][:-4]

    try:
        if os.path.exists(report_csv_dir):
            csv_record, page_count = unzip(pdf_path, csv_dir, cache=True)
        else:
            zip_path = extract_table(pdf_path, target_dir, execution_context)
            csv_record, page_count = unzip(zip_path, csv_dir)
    except ErrorCode as e:
        return e, None, None
    
    return None, csv_record, page_count


