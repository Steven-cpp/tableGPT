from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException

from dotenv import load_dotenv
import os
import pandas as pd
import time
import zipfile
import logging
from tqdm import tqdm


def handle_exception(exception_type, exception_message, status_code):
    logging.info(exception_type)
    if status_code is not None:
        logging.info(status_code)
    logging.info(exception_message)


# Extract tables from pdfs under given dir
def extract_table(directory: str, cont: ExecutionContext):
    
    for file in tqdm(os.listdir(directory), desc='PDF Table Extraction'):
        try:
            start = time.time()
            fn = os.fsdecode(file)
            if not fn.endswith('pdf'):
                continue
            logging.info('Extracting tables from %s', fn)
            #Set operation input from a source file.
            source = FileRef.create_from_local_file(os.path.join(directory, fn))
            extract_pdf_operation = ExtractPDFOperation.create_new()
            extract_pdf_operation.set_input(source)

            #Build ExtractPDF options and set them into the operation
            extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
                .with_element_to_extract(ExtractElementType.TABLES)\
                .with_table_structure_format(TableStructureType.CSV) \
                .build()

            extract_pdf_operation.set_options(extract_pdf_options)

            #Execute the operation.
            result: FileRef = extract_pdf_operation.execute(cont)
            zip_file = f"./tmp/{fn[:-4]}.zip"
            #Save the result to the specified location.
            result.save_as(zip_file)
            end = time.time()
            
            print('Table extraction of %s finishes: %d sec.', fn, int(end - start))

        except ServiceApiException as service_api_exception:
            # ServiceApiException is thrown when an underlying service API call results in an error.
            handle_exception("ServiceApiException", service_api_exception.message, service_api_exception.status_code)

        except ServiceUsageException as service_usage_exception:
            # ServiceUsageException is thrown when either service usage limit has been reached or credentials quota has been
            # exhausted.
            handle_exception("ServiceUsageException", service_usage_exception.message, service_usage_exception.status_code)

        except SdkException as sdk_exception:
            # SdkException is typically thrown for client-side or network errors.
            handle_exception("SdkException", sdk_exception.message, None)
    
    
# Unzip the package obtained via API and rename it
def unzip(src: str, dest: str) -> int:
    for file in os.listdir(src):
        zipFn = os.fsdecode(file)
        if not zipFn.endswith('zip'):
            continue
        zip_path = os.path.join(src, zipFn)
        ext_dir = os.path.join(src, zipFn[:-4])
        with zipfile.ZipFile(zip_path, 'r') as zipFile:
            zipFile.extractall(ext_dir)
        os.remove(zip_path)
        if not os.path.exists(os.path.join(ext_dir, 'tables')):
            logging.warning(f'No Table Extracted from {ext_dir}')
            continue
        
        dest_dir = os.path.join(dest, f'{zipFn[:-4]}')

        try:
            os.mkdir(dest_dir)
        except OSError as e:
            print(e)
            return 0
        
        file_num = len(os.listdir(os.path.join(ext_dir, 'tables')))
        for idx, tab in enumerate(sorted(os.listdir(os.path.join(ext_dir, 'tables')))):
            os.rename(os.path.join(ext_dir, 'tables', tab), \
                      os.path.join(dest_dir, f'{zipFn[:-4]}_{idx}.csv'))
        os.rename(os.path.join(ext_dir, 'structuredData.json'),
                    os.path.join(dest_dir, 'structuredData.json'))

        logging.info(f'{zipFn} extraction finished')
        return file_num


def extract_pdf(pdf_dir: str, csv_dir: str) -> int:
    load_dotenv('.env')

    # Initial setup, create credentials instance.
    credentials = Credentials.service_principal_credentials_builder()\
        .with_client_id(os.getenv('PDF_SERVICES_CLIENT_ID'))\
        .with_client_secret(os.getenv('PDF_SERVICES_CLIENT_SECRET')).build()

    # Create an ExecutionContext using credentials and create a new operation instance.
    execution_context = ExecutionContext.create(credentials)
    
    extract_table(pdf_dir, execution_context)
    return unzip('./tmp', csv_dir)


