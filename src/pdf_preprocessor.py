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
import json
import os

sit_keywords = ['schedule of investments', 'schedule of portfolio investments', 'investment schedule']
pst_keywords = ['portfolio summary', 'active portfolio', 'investments currently in the portfolio', 'investments as of',
                'investment multiple and gross irr', 'portfolio company summary', 'investment performance']


def __is_continuation(x1: list, x2: list, n: int) -> bool:
    """ Check if x2 is the continuation of x1 by checking x axis matches

    Args:
        x1 (list): x axis of the last 20 bboxes
        x2 (list): x axis of the first 20 bboxes
        n (int): number of matches requried

    Returns:
        bool:
         * True: more than `n` pairs of x1-x2 matches
         * False: less than `n` pairs of x1-x2 matches
    """
    x1.sort()
    x2.sort()
    i, j, cnt = 0, 0, 0

    while i < len(x1):
        if j > len(x2) - 1:
            break
        if abs(x1[i] - x2[j]) < 1:
            cnt += 1
            j += 1
            i += 1
        elif x1[i] > x2[j]:
            j += 1
        else:
            i += 1
            continue
    return True if cnt >= n else False



def __contain_pst_keywords(page, rule_config, n=3) -> bool:
    """ Check whether the top of the page contains keywords that may construct PST

    Args:
        page (page): the page object to be checked
        rule_config (json): the rule defined by `config.json`
        n (int, optional): the least number of keywords to be matched. Defaults to 4.

    Returns:
        bool: whether the page contains sufficient keywords
            * True: there are sufficient keywords, should further check
            * False: keywords not sufficient, ignore this page
    """
    target_metrics = ['total_cost', 'unrealized_value', 'realized_value', 'total', 'gross_moic']
    mask = [False] * len(rule_config)
    top_at = 1/6

    # Capture all the texts within the `top_at` region
    text = ''
    text_blocks = page.get_text('dict')['blocks']
    for block in text_blocks:
        if "bbox" in block and block['bbox'][1] > page.rect.height * top_at:
            break
        if "lines" in block:
            for line in block['lines']:
                if not (abs(line['dir'][0] - 1) < 0.01 or abs(line['dir'][0] - 0) < 0.01 or abs(line['dir'][0] + 1) < 0.01):
                    continue
                for span in line["spans"]:
                    # Check for watermark properties
                    if "confidential" not in span["text"].lower() and "@sofinagroup" not in span["text"].lower() and span["size"] < 20:
                        # Add the span text to the new page
                        text += span["text"]
    text_lower = text.lower()

    for idx, metric in enumerate(rule_config):
        if metric not in target_metrics:
            continue
        rule = rule_config[metric]
        if 'ColumnNamePattern' not in rule:
            continue
        namePatterns = rule['ColumnNamePattern']
        for rule in namePatterns:
            if 'isRegex' in rule:
                continue
            patterns = rule['Patterns']
            for pat in patterns:
                if len(pat.split()) > 1 and pat.lower() in text_lower:
                    mask[idx] = True
                    continue
    return True if sum(mask) >= n else False


def update_map(map_obj, **args):
    for key, value in args.items():
        if key in map_obj:
            map_obj[key].append(value)
        else:
            raise ValueError(f'Key {key} does not exist in the given dict')

def process_docs(report_paths: list, rule_path, output_dir='./output', fn='report_nowm'):
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
    rule_config = json.load(open(rule_path))
    for path in report_paths:
        try:
            doc = pymupdf.open(path)
        except Exception as e:
            logging.warning(f'Failed to read the report: {e}.')
            continue
        cnt_page_lst = cnt_page
        last_page_xs = None
        for id, page in enumerate(doc):
            table_type, current_xs = process_page(page, new_doc, rule_config, last_page_xs)
            last_page_xs = current_xs
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


def process_page(page, new_doc, rule_config, last_page_xs):
    page_text_lower = page.get_text().lower().replace('\n', '')
    table_type = ''
    check_continuation = False
    current_page_cols = -1
    if any(keyword in page_text_lower for keyword in sit_keywords):
        table_type = 'SIT'
    elif any(keyword in page_text_lower for keyword in pst_keywords):
        table_type = 'PST'
    elif (__contain_pst_keywords(page, rule_config, n=3)):
        table_type = 'PST'
    elif last_page_xs:
        table_type = 'PST'
        check_continuation = True
    else:
        return '', None
    
    if len(page_text_lower) < 200:
        return '', None

    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
    is_rotated_90 = (page.rotation == 90)
    # Get all text blocks on the page
    text_blocks = page.get_text("dict")["blocks"]
    span_xs, span_txts = [], []
    
    for block in text_blocks:
        # Filter for text lines
        if "lines" in block:
            for line in block["lines"]:
                if not (abs(line['dir'][0] - 1) < 0.01 or abs(line['dir'][0] - 0) < 0.01 or abs(line['dir'][0] + 1) < 0.01):
                    continue
                for span in line["spans"]:
                    # Check for watermark properties
                    if "confidential" not in span["text"].lower() and "@sofinagroup" not in span["text"].lower() and span["size"] < 20:
                        # Add the span text to the new page
                        span_xs.append(span["bbox"][0])
                        rec = pymupdf.Rect(span["bbox"][0], span["bbox"][1], span["bbox"][2], span["bbox"][3])
                        span_txts.append(span["text"].lower())
                        if is_rotated_90:
                            rec[0] = min(rec[0], rec[2])
                        elif rec.height > rec.width * 1.85:
                             new_page.insert_text((rec[2] - 2, rec[3] - 8), span["text"].replace('$', ''),
                                             fontsize=span["size"]-0.5, color=(0, 0, 0), rotate=90)
                             continue
                        rec = rec * page.rotation_matrix
                        new_page.insert_text((rec[0], rec[1]), span["text"].replace('$', ''),
                                             fontsize=span["size"]-0.5, color=(0, 0, 0))

    span_xs_top, span_xs_tail = list(set(span_xs[:32])), list(set(span_xs[-32:]))
    # span_txts_top, span_txts_tail = span_txts[:32], span_txts[-32:]
    if last_page_xs and check_continuation and not __is_continuation(span_xs_top, last_page_xs, n=5):
        page_index = new_doc.page_count - 1
        new_doc.delete_page(page_index)
        return '', None        


    return table_type, span_xs_tail
