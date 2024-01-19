from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType

import os.path
import zipfile
import json

zip_file = "./ExtractTableFromPDF.zip"

if os.path.isfile(zip_file):
    os.remove(zip_file)

input_pdf = "./CA_Sample1.pdf"

os.environ["PDF_SERVICES_CLIENT_ID"] = "f66a646af0374637b0cca91b42472eab"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-Ho9aaHwP3nlgk23r0LYBpxjMhnuQWsJX"

#Initial setup, create credentials instance.
credentials = Credentials.service_principal_credentials_builder().with_client_id(os.getenv('PDF_SERVICES_CLIENT_ID')).with_client_secret(os.getenv('PDF_SERVICES_CLIENT_SECRET')).build()

#Create an ExecutionContext using credentials and create a new operation instance.
execution_context = ExecutionContext.create(credentials)

extract_pdf_operation = ExtractPDFOperation.create_new()

#Set operation input from a source file.
source = FileRef.create_from_local_file(input_pdf)
extract_pdf_operation.set_input(source)

#Build ExtractPDF options and set them into the operation
extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
    .with_element_to_extract(ExtractElementType.TABLES) \
    .build()
extract_pdf_operation.set_options(extract_pdf_options)

#Execute the operation.
result: FileRef = extract_pdf_operation.execute(execution_context)

#Save the result to the specified location.
result.save_as(zip_file)


archive = zipfile.ZipFile(zip_file, 'r')
jsonentry = archive.open('structuredData.json')
jsondata = jsonentry.read()
data = json.loads(jsondata)

for element in data["elements"]:
    if(element["Path"].endswith("/H1")):
        print(element["Text"])



