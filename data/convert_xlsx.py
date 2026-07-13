import zipfile
import xml.etree.ElementTree as ET
import csv
import re
import os

def col_to_num(col_str):
    num = 0
    for c in col_str:
        if c.isalpha():
            num = num * 26 + (ord(c.upper()) - ord('A') + 1)
    return num - 1

def parse_cell_ref(ref):
    match = re.match(r"([A-Z]+)([0-9]+)", ref)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def get_shared_strings(z):
    try:
        xml_content = z.read('xl/sharedStrings.xml')
        root = ET.fromstring(xml_content)
        ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        strings = []
        for si in root.findall('.//ns:si', ns):
            t_texts = [t.text for t in si.findall('.//ns:t', ns) if t.text]
            strings.append("".join(t_texts) if t_texts else "")
        return strings
    except KeyError:
        return []

def convert_xlsx_to_csv(xlsx_path, csv_path):
    print(f"Opening zip file {xlsx_path}...")
    with zipfile.ZipFile(xlsx_path, 'r') as z:
        strings = get_shared_strings(z)
        print(f"Loaded {len(strings)} shared strings.")

        try:
            xml_content = z.read('xl/worksheets/sheet1.xml')
        except KeyError:
            print("Error: Could not find sheet1.xml")
            return
        
        print("Parsing sheet1.xml...")
        root = ET.fromstring(xml_content)
        ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        
        rows = root.findall('.//ns:row', ns)
        print(f"Found {len(rows)} rows.")
        
        max_col_idx = 0
        parsed_rows = {}
        
        for r_elem in rows:
            row_num = int(r_elem.get('r'))
            cells = r_elem.findall('ns:c', ns)
            row_data = {}
            for cell in cells:
                ref = cell.get('r')
                col_letter, _ = parse_cell_ref(ref)
                if col_letter:
                    col_idx = col_to_num(col_letter)
                    max_col_idx = max(max_col_idx, col_idx)
                    
                    val_elem = cell.find('ns:v', ns)
                    val = val_elem.text if val_elem is not None else ""
                    t = cell.get('t')
                    if t == 's' and val:
                        try:
                            str_idx = int(val)
                            if str_idx < len(strings):
                                val = strings[str_idx]
                        except ValueError:
                            pass
                    row_data[col_idx] = val
            parsed_rows[row_num] = row_data
            
        print(f"Max column index: {max_col_idx} (total {max_col_idx+1} columns)")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            max_row_num = max(parsed_rows.keys()) if parsed_rows else 0
            for r in range(1, max_row_num + 1):
                row_data = parsed_rows.get(r, {})
                row_list = [row_data.get(c, "") for c in range(max_col_idx + 1)]
                writer.writerow(row_list)
                
        print(f"Successfully converted and saved to {csv_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx = os.path.join(base_dir, "data", "raw", "tilapia_iot.xlsx")
    csv_out = os.path.join(base_dir, "data", "raw", "tilapia_iot.csv")
    convert_xlsx_to_csv(xlsx, csv_out)
