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
import hashlib
import logging
import os

sit_keywords = ['schedule of investments', 'schedule of portfolio investments', 'investment schedule']
pst_keywords = ['portfolio summary', 'active portfolio', 'investments currently in the portfolio', 'investments as of',
                'investment multiple and gross irr']

def update_map(map_obj, **args):
    for key, value in args.items():
        if key in map_obj:
            map_obj[key].append(value)
        else:
            raise ValueError(f'Key {key} does not exist in the given dict')

def process_docs(report_paths: list, output_dir='./output', fn='report_nowm'):
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
        try:
            doc = pymupdf.open(path)
        except Exception as e:
            logging.warning(f'Failed to read the report: {e}.')
            continue
        cnt_page_lst = cnt_page
        for id, page in enumerate(doc):
            table_type = process_page(page, new_doc)
            if table_type == '':
                continue
            update_map(doc_map, report_path=path, report_name=path.split('/')[-1],
                       page_ori=id+1, page_new=cnt_page+1, table_type=table_type,
                       is_processed=True)
            cnt_page += 1
        # If no SIT or PST tables are identified, also add this report as a reference
        if cnt_page_lst == cnt_page:
            update_map(doc_map, report_path=path, report_name=path.split('/')[-1],
                       page_ori=None, page_new=None, table_type=None, is_processed=False)
    if cnt_page == 0:
        return None, None
    uuid = hashlib.sha256(','.join(report_paths).encode('utf-8')).hexdigest()[:4]
    output_path = os.path.join(output_dir, f'{fn}_{uuid}.pdf')
    new_doc.save(output_path)
    return output_path, doc_map


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
                    if "confidential" not in span["text"].lower() and "@sofinagroup" not in span["text"].lower() and span["size"] < 20:  # Customize this condition
                        # Add the span text to the new page
                        new_page.insert_text((span["bbox"][0], span["bbox"][1]), span["text"].replace('$', ''),
                                             fontsize=span["size"], color=(0, 0, 0))
    
    return table_type
