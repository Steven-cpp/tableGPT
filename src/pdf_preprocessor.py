"""
Combine GP reports into one pdf file with only non-watermark text blocks of pages
containing Schedule of Investments Table (SIT) and Porfolio Summary Table (PST)

Main Logic
----------
This script reads a list of PDF files, iterating each page to check whether
it contains SIT or PST:
    * Contains: filter out watermark text blocks, and insert all other text blocks
      into a new page.
    * NOT Contains: ignore this page

Output
------
    `./output/portco_tables_nowm.pdf`: a recreated new pdf file that combines SIT
                                       and PST from various GP reports.
    doc_map (dict): the origianl pdf metadata
"""
import pymupdf
import os

sit_keywords = ['schedule of investments', 'schedule of portfolio investments', 'investment schedules']
pst_keywords = ['portfolio summary']

def process_docs(report_paths: list, output_dir='./output'):
    doc_map = {
        'report_path': [],
        'report_name': [],
        'page_ori': [],
        'page_new': [],
        'table_type': [],
        'is_processed': []
    }
    cnt_page = 0
    new_doc = pymupdf.open()
    for path in report_paths:
        doc = pymupdf.open(path)
        cnt_page_lst = cnt_page
        for id, page in enumerate(doc):
            table_type = process_page(page, new_doc)
            if table_type == '':
                continue
            doc_map['report_path'].append(path)
            doc_map['report_name'].append(path.split('/')[-1])
            doc_map['page_ori'].append(id + 1)
            doc_map['page_new'].append(cnt_page + 1)
            doc_map['table_type'].append(table_type)
            doc_map['is_processed'].append(True)
            cnt_page += 1
        # If no SIT or PST tables are identified, also add this report as a reference
        if cnt_page_lst == cnt_page:
            doc_map['report_path'].append(path)
            doc_map['report_name'].append(path.split('/')[-1])
            doc_map['page_ori'].append(None)
            doc_map['page_new'].append(None)
            doc_map['table_type'].append(None)
            doc_map['is_processed'].append(False)

    new_doc.save(os.path.join(output_dir, 'portco_tables_nowm.pdf'))
    return doc_map


def process_page(page, new_doc):
    page_text_lower = page.get_text().lower()
    table_type = ''
    if any(keyword in page_text_lower for keyword in sit_keywords):
        table_type = 'SIT'
    elif any(keyword in page_text_lower for keyword in pst_keywords):
        table_type = 'PST'
    else:
        return ''
    
    if len(page_text_lower) < 250:
        return ''

    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
    # Get all text blocks on the page
    text_blocks = page.get_text("dict")["blocks"]
    
    for block in text_blocks:
        # Filter for text lines
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Check for watermark properties
                    if "confidential" not in span["text"].lower() and "finance-pe" not in span["text"].lower():  # Customize this condition
                        # Add the span text to the new page
                        new_page.insert_text((span["bbox"][0], span["bbox"][1]), span["text"],
                                             fontsize=span["size"], color=(0, 0, 0))
    
    return table_type