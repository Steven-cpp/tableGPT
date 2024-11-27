"""_summary_


Example Usage
-------------
def process_file(file):
    if file.size > 50 * 1024 * 1024:
        raise FileSizeError()
    if file.page_count > 200:
        raise PageCountError()
try:
    process_file(some_file)
except ErrorCode as e:
    print(e)  # This will print the error code, level, and description

"""

class ErrorCode(Exception):
    def __init__(self, code, level, description):
        self.code = code
        self.level = level
        self.description = description
    
    def __str__(self):
        return f"[{self.code}] {self.level}: {self.description}"

class SPReportDownloadError(ErrorCode):
    def __init__(self, message):
        super().__init__("ERR-RP1001", "Error", f"Failed to download this report from SharePoint: {message}.")

class InvalidReportReadError(ErrorCode):
    def __init__(self, report_path):
        super().__init__("ERR-RP1002", "Error", f"Failed to read the report: {report_path}.")


class FileSizeError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-EX1001", "Error", "File size is larger than 50MB")

class PageCountError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-EX1002", "Error", "Page count is more than 200")

class AdobeAPIRequestError(ErrorCode):
    def __init__(self, message, status_code):
        super().__init__("ERR-EX1003", "Error", f"Adobe PDF Extract API exception occurred: {message}.Status Code: {status_code}")

class NoTableExtractedWarning(ErrorCode):
    def __init__(self):
        super().__init__("WRN-EX1004", "Warning", "No table extracted from the given pdf report")

class UnzipError(ErrorCode):
    def __init__(self, message):
        super().__init__("ERR-EX1005", "Error", f"Failed to unzip the file from PDF Extract API: {message}")

class CreateAzureServiceError(ErrorCode):
    def __init__(self, message):
        super().__init__("ERR-EX1006", "Error", f"Failed to create ADI service: {message}")

class ReportNotFoundError(ErrorCode):
    def __init__(self, path):
        super().__init__("ERR-EX1007", "Error", f"Failed to find the specified report: {path}")

class AnalyzeDocumentError(ErrorCode):
    def __init__(self, message):
        super().__init__("ERR-EX1008", "Error", f"Failed to analyze the document: {message}")

class NoTablePageDetectedWarning(ErrorCode):
    def __init__(self, path):
        super().__init__("WRN-EX1009", "Warning", f"Cannot find page containing SIT or PST from the given report: {path}")

class UnsupportedMergedCellError(ErrorCode):
    def __init__(self, cotent):
        super().__init__("ERR-EX1010", "Error", f"Cannot process merged cell spanning more than 2 rows: {cotent}")

class InvalidCSVError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-ID1001", "Error", "The csv file is invalid")

class UnsupportedReportTypeWarning(ErrorCode):
    def __init__(self):
        super().__init__("WRN-ID2001", "Warning", "The report type is unsupported, maybe split by series")

class InvalidTableWarning(ErrorCode):
    def __init__(self):
        super().__init__("WRN-ID2002", "Warning", "Less than 3 columns identified, table is invalid")

class P1RuleMultiMatchError(ErrorCode):
    def __init__(self, msg):
        super().__init__("ERR-ID1002", "Error", f"P1 rule {msg} has multi-matches")

class P2RuleMultiMatchWarning(ErrorCode):
    def __init__(self, msg):
        super().__init__("WRN-ID2003", "Warning", f"P2 rule {msg} has multi-matches")

class UnsupportedC2ReportTypeWarning(ErrorCode):
    def __init__(self, message):
        super().__init__("WRN-ID2004", "Warning", f"The schedule of investments table of type C2 is unsupported: {message}")
class InvalidConfigError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-ID1003", "Error", "Missing required configs, ['Patterns', 'Method', 'Priority'] are required, config is invalid")

class InvalidMethodError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-ID1004", "Error", "Invalid method in rule, config is invalid")

class UnsupportedReportSchemaError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-ID1005", "Error", "The extractor does not support this csv file that should be identified")

class InvalidC1ReportSchemaError(ErrorCode):
    def __init__(self):
        super().__init__("ERR-ID1006", "Error", "The schedule of investments table of type C1 is invalid, does not align with the expected schema.")

class NullTableAfterTransformationWarning(ErrorCode):
    def __init__(self):
        super().__init__("WRN-TR2001", "Warning", "Table is null after transformation")

class ValidationFailedWarning(ErrorCode):
    def __init__(self):
        super().__init__("WRN-VA2001", "Warning", "Table validation failed, 'UnrealizeValue' + 'RealizedValue' = 'Total' NOT Passed")
