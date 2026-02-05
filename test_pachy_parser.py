import re

def parse_pachy_metrics(text):
    """
    Specific parser for the Pachymetry panel which has a grid-like or multi-line structure.
    Handles labels followed by value, X, and Y on subsequent lines.
    """
    data = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Map keywords to base column names
    labels_config = [
        {'id': 'PupilCenter', 'keywords': ['Puc', 'Pupil']},
        {'id': 'PachyApex', 'keywords': ['Pachy Vertex', 'Pachy Apex', 'Pachy Vert']},
        {'id': 'ThinnestLoc', 'keywords': ['Thinnest']},
        {'id': 'KMax', 'keywords': ['K Max', 'KMax']}
    ]
    
    def extract_num(txt):
        if not txt: return None
        txt_clean = txt.replace('~', '-')
        match = re.search(r'([+\-]?\d+\.?\d*)', txt_clean)
        return match.group(1) if match else None

    for i, line in enumerate(lines):
        matched_metric = None
        for cfg in labels_config:
            if any(k.lower() in line.lower() for k in cfg['keywords']):
                matched_metric = cfg['id']
                break
        
        if matched_metric:
            current_line_parts = line.split(':', 1)
            current_line_val = extract_num(current_line_parts[1] if len(current_line_parts) > 1 else "")
            
            look_ahead_idx = i + 1
            results = []
            
            if current_line_val:
                results.append(current_line_val)
                
            while len(results) < 3 and look_ahead_idx < len(lines):
                next_line = lines[look_ahead_idx]
                
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
            
            if len(results) >= 1:
                data[matched_metric] = results[0]
            if len(results) >= 2:
                data[f"{matched_metric}_X"] = results[1]
            if len(results) >= 3:
                data[f"{matched_metric}_Y"] = results[2]
                
    return data

test_text = """
Puc" Center:
453 pm
~0.02
+0.01
Pachy Vertex N._
452 pm
0.00
0.00
Thinnest Locat:
448 pm
+0.78
-0.26
K Max (Front}:
485D
-0.20
-0.78
"""

result = parse_pachy_metrics(test_text)
print(result)

# Expected outputs check
assert result['PupilCenter'] == '453'
assert result['PupilCenter_X'] == '-0.02'
assert result['PupilCenter_Y'] == '+0.01'
assert result['PachyApex'] == '452'
assert result['PachyApex_X'] == '0.00'
assert result['PachyApex_Y'] == '0.00'
assert result['ThinnestLoc'] == '448'
assert result['ThinnestLoc_X'] == '+0.78'
assert result['ThinnestLoc_Y'] == '-0.26'
assert result['KMax'] == '485'
assert result['KMax_X'] == '-0.20'
assert result['KMax_Y'] == '-0.78'
print("Test Passed!")
