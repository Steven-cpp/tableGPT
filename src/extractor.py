from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType

from dotenv import load_dotenv
import os
import pandas as pd
import time
import zipfile


# Extract tables from pdfs under given dir
def extract_table(directory: str, cont: ExecutionContext):
    
    for idx, file in enumerate(os.listdir(directory)):
        start = time.time()
        fn = os.fsdecode(file)
        if not fn.endswith('pdf'):
            continue
        print('*'*8 + f' {idx}: Extracting tables from {fn} ' + '*'*8)
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
        zip_file = f"./res/{fn[:-4]}.zip"
        #Save the result to the specified location.
        result.save_as(zip_file)
        end = time.time()
        
        print(f'Table extraction of {fn} finishes: {int(end - start)} sec.')
    
    
# Unzip the package obtained via API and rename it
def unzip(src: str):
    dest_csv = './res/final'
    for file in os.listdir(src):
        zipFn = os.fsdecode(file)
        if not zipFn.endswith('zip'):
            continue
        zip_path = os.path.join(src, zipFn)
        ext_path = f'./res/{zipFn[:-4]}/'
        with zipfile.ZipFile(zip_path, 'r') as zipFile:
            zipFile.extractall(ext_path)
        os.remove(zip_path)
        if not os.path.exists(os.path.join(ext_path, 'tables')):
            print(f'No Table Extracted from {ext_path}')
            continue
        
        for idx, tab in enumerate(sorted(os.listdir(os.path.join(ext_path, 'tables')))):
            os.rename(os.path.join(ext_path, 'tables', tab), os.path.join(dest_csv, f'{zipFn[:-4]}_{idx}.csv'))

        print(f'{zipFn} extraction finished')
    
    


if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    os.environ["PDF_SERVICES_CLIENT_ID"] = "f66a646af0374637b0cca91b42472eab"
    os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-Ho9aaHwP3nlgk23r0LYBpxjMhnuQWsJX"

    # Initial setup, create credentials instance.
    credentials = Credentials.service_principal_credentials_builder().with_client_id(os.getenv('PDF_SERVICES_CLIENT_ID')).with_client_secret(os.getenv('PDF_SERVICES_CLIENT_SECRET')).build()

    # Create an ExecutionContext using credentials and create a new operation instance.
    execution_context = ExecutionContext.create(credentials)
    
    extract_table('./docs', execution_context)
    unzip('./res')



