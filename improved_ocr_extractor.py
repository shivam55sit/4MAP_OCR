import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import os

# Initialize EasyOCR Reader (loads model once)
reader = easyocr.Reader(['en'])

# Set Tesseract path explicitly (if needed as fallback, but using EasyOCR primarily)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shivam.prajapati\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def split_panels(image):
    """
    Splits the stitched image into sub-panels based on white space separators.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold to find white rows (separators are pure white 255)
    row_means = np.mean(gray, axis=1)
    is_white_row = row_means > 250

    rows = []
    start_idx = None
    
    for i, is_white in enumerate(is_white_row):
        if not is_white:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                rows.append((start_idx, i))
                start_idx = None
    
    if start_idx is not None:
        rows.append((start_idx, len(gray)))

    panels = []
    for start, end in rows:
        if (end - start) > 20: 
            panels.append(image[start:end, :])
            
    return panels

def extract_text(image):
    """Run EasyOCR on an image crop."""
    # detail=0 returns just the list of text strings
    # paragraph=False might be better to get line-by-line for key-value parsing
    result = reader.readtext(image, detail=0, paragraph=True) 
    # Joining with newlines to form a document-like string
    return '\n'.join(result)

def clean_val(val):
    """Minimal cleaning: just strip whitespace."""
    if val:
        return val.strip()
    return val

# ============================================================================
# DYNAMIC PARSERS
# ============================================================================

def parse_key_value_pairs(text, prefix=""):
    """
    Parses 'Label: Value' or 'Label; Value' pairs from text.
    Handles multi-line values and extracts standalone ID patterns.
    """
    data = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # regex for ID (starts with P followed by digits)
    id_pattern = r'(P\d{6,})'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for delimiters : or ;
        match = re.search(r'[:;]', line)
        if match:
            delimiter_pos = match.start()
            key = line[:delimiter_pos].strip()
            val = line[delimiter_pos+1:].strip()
            
            # Key normalization
            key_clean = re.sub(r'[^\w\s]', '', key).strip()
            
            # Multi-line value check: if current value is empty, look at next line
            if not val and i + 1 < len(lines):
                next_line = lines[i+1]
                if ':' not in next_line and ';' not in next_line:
                    val = next_line
                    i += 1
            
            # Special case: Value might contain the ID merged into it (e.g. "Shafiya P1665127")
            id_match = re.search(id_pattern, val or "", re.IGNORECASE)
            if id_match:
                extracted_id = id_match.group(1)
                data[f"{prefix}ID"] = extracted_id
                # Remove ID from the current value
                val = val.replace(extracted_id, "").strip()
            
            if key_clean:
                data[f"{prefix}{key_clean}"] = val
        else:
            # Standalone line check for ID
            # Also handles case where ID is on its own line after a key
            id_match = re.search(id_pattern, line, re.IGNORECASE)
            if id_match:
                data[f"{prefix}ID"] = id_match.group(1)
            elif i > 0:
                # Potential value for previous key if it was empty
                prev_key = list(data.keys())[-1] if data else None
                if prev_key and not data[prev_key]:
                    data[prev_key] = line
                    
        i += 1

    return data

def parse_pachy_metrics(text):
    """
    Specific parser for the Pachymetry panel which has a grid-like or multi-line structure.
    Handles labels followed by value, X, and Y on subsequent lines.
    """
    data = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Map keywords to base column names
    # Using small fragments for robust matching against OCR noise
    labels_config = [
        {'id': 'PupilCenter', 'keywords': ['Puc', 'Pupil']},
        {'id': 'PachyApex', 'keywords': ['Pachy Vertex', 'Pachy Apex', 'Pachy Vert']},
        {'id': 'ThinnestLoc', 'keywords': ['Thinnest']},
        {'id': 'KMax', 'keywords': ['K Max', 'KMax']}
    ]
    
    def extract_num(txt):
        if not txt: return None
        # Map OCR artifacts: '~' often stands for '-'
        txt_clean = txt.replace('~', '-')
        # Handles artifacts like '+', '-', 'um', 'pm', 'D'
        # Pattern looks for any digits/dots/signs
        match = re.search(r'([+\-]?\d+\.?\d*)', txt_clean)
        return match.group(1) if match else None

    for i, line in enumerate(lines):
        matched_metric = None
        for cfg in labels_config:
            if any(k.lower() in line.lower() for k in cfg['keywords']):
                matched_metric = cfg['id']
                break
        
        if matched_metric:
            # We found a label. Now look for Val, X, Y in the following lines
            # Sometimes the value is on the SAME line as the label (e.g. "K Max: 48.5 D")
            # If not, look ahead.
            
            # Check if there's a numeric value on the current line
            # (But skip if the label itself contains digits like "K1")
            # For pachy, usually labels don't have digits we want to capture
            current_line_val = extract_num(line.split(':', 1)[1] if ':' in line else "")
            
            look_ahead_idx = i + 1
            results = []
            
            # If we found a value on current line, count it
            if current_line_val:
                results.append(current_line_val)
                
            # Grab up to 3 numeric values from subsequent lines
            while len(results) < 3 and look_ahead_idx < len(lines):
                next_line = lines[look_ahead_idx]
                
                # If we hit another known label, stop looking ahead
                is_another_label = False
                for cfg in labels_config:
                    if any(k.lower() in next_line.lower() for k in cfg['keywords']):
                        is_another_label = True
                        break
                if is_another_label:
                    break
                    
                val = extract_num(next_line)
                if val:
                    results.append(val)
                look_ahead_idx += 1
            
            # Map results to data
            if len(results) >= 1:
                data[matched_metric] = results[0]
            if len(results) >= 2:
                data[f"{matched_metric}_X"] = results[1]
            if len(results) >= 3:
                data[f"{matched_metric}_Y"] = results[2]
                
    return data


def extract_metrics_from_cropped(image_input):
    """
    Main function to process the cropped panel(s).
    Args:
        image_input (str OR dict): Path to a stacked image OR dict of {name: np_array}
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
            
        full_img = cv2.imread(image_input)
        if full_img is None:
            raise ValueError("Could not read image")
            
        panels_list = split_panels(full_img)
        # Convert list to dict for unified processing if possible, or mapping
        # In the old logic, we assumed order: 0:pat, 1:cf, 2:cb, 3:pachy, 4:others
        panels = {}
        names = ['pat_info', 'cornea_front', 'cornea_back', 'pachy', 'others']
        for i, img in enumerate(panels_list):
            if i < len(names):
                panels[names[i]] = img
        
        image_name = os.path.basename(image_input)
    elif isinstance(image_input, dict):
        panels = image_input
        image_name = "individual_crops"
    else:
        raise ValueError("Input must be a file path string or a dictionary of images")
        
    extracted_data = {}
    extracted_data['image_name'] = image_name
    
    # 1. Patient Info
    if 'pat_info' in panels:
        txt = extract_text(panels['pat_info'])
        print(f"--- PATIENT INFO RAW ---\n{txt}\n-------------------")
        extracted_data.update(parse_key_value_pairs(txt))
        
    # 2. Cornea Front - Prefix 'cf_'
    if 'cornea_front' in panels:
        txt = extract_text(panels['cornea_front'])
        print(f"--- CORNEA FRONT RAW ---\n{txt}\n-------------------")
        extracted_data.update(parse_key_value_pairs(txt, prefix='cf_'))
        
    # 3. Cornea Back - Prefix 'cb_'
    if 'cornea_back' in panels:
        txt = extract_text(panels['cornea_back'])
        print(f"--- CORNEA BACK RAW ---\n{txt}\n-------------------")
        extracted_data.update(parse_key_value_pairs(txt, prefix='cb_'))
        
    # 4. Pachy - Structured Grid
    if 'pachy' in panels:
        txt = extract_text(panels['pachy'])
        print(f"--- PACHY RAW ---\n{txt}\n-------------------")
        extracted_data.update(parse_pachy_metrics(txt))
        
    # 5. Global Metrics
    if 'others' in panels:
        txt = extract_text(panels['others'])
        print(f"--- OTHERS RAW ---\n{txt}\n-------------------")
        extracted_data.update(parse_key_value_pairs(txt))
    
    return extracted_data

def process_and_save(image_path, output_excel):
    data = extract_metrics_from_cropped(image_path)
    
    # Create DataFrame dynamically from keys found
    df = pd.DataFrame([data])
    
    # Sort columns for better readability? 
    # Put image_name first, then Patient info, then cf_, cb_, pachy, others
    cols = list(df.columns)
    
    # Helper to sort
    def sort_key(col):
        if col == 'image_name': return '000'
        if col.startswith('cf_'): return '200' + col
        if col.startswith('cb_'): return '300' + col
        # Heuristic for patient info: usually starts with standard names
        if col in ['First Name', 'Last Name', 'ID', 'DOB', 'Date of Birth', 'Exam Date', 'Eye', 'Time']:
            return '100' + col
        return '900' + col # Others last
        
    sorted_cols = sorted(cols, key=sort_key)
    df = df[sorted_cols]
    
    df.to_excel(output_excel, index=False)
    print(f"Saved results to {output_excel}")
    print("Columns found:", sorted_cols)

if __name__ == "__main__":
    # Test path
    img_path = r"C:\Users\shivam.prajapati\Desktop\4MAP_OCR\crops\pat_info.jpg"
    out_path = r"C:\Users\shivam.prajapati\Desktop\4MAP_OCR\extracted_results.xlsx"
    
    if os.path.exists(img_path):
        process_and_save(img_path, out_path)
    else:
        print("Image not found. Run image_crop.py first.")