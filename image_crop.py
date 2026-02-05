import cv2
import numpy as np
import os
import re
from constants import IMG_SIZE_TO_DIMENSIONS, used_img_dims, model_headers, image_header_coord_dict

# Region coordinates extracted from functions.py
# Format: (y1, y2, x1, x2)
# White regions: [(y1, y2, x1, x2), ...] relative to the cropped region or absolute? 
# In functions.py: regions[region][y1:y2, x1:x2] = 255. 
# The white_regions coordinates in functions.py seem to be relative to the *cropped region* because 
# the code does `regions[region][y1:y2, x1:x2] = 255`. 
# Let's verify: 
# (758, 1200) 'cornea_front': (188, 353, 88, 295) -> height 165, width 207
# 'white_regions': 'cornea_front': [(49, 64, 0, 5)] -> y=49-64, x=0-5. This fits inside the cropped region.
# So white_regions are relative to the top-left of the specific crop.

REGION_COORDS_OCULYZER = {
    (758, 1200): {
        'pat_info': (43, 173, 10, 295),
        'cornea_front': (188, 353, 88, 295),
        'cornea_back': (357, 521, 87, 295),
        'pachy': (541, 639, 10, 295),
        'others': (642, 752, 10, 295),
        'white_regions': {
            'cornea_front': [(49, 64, 0, 5)],
            'cornea_back': [(49, 64, 0, 5)]
        }
    },
    (820, 1200): {
        'pat_info': (45, 186, 10, 321),
        'cornea_front': (208, 371, 97, 317),
        'cornea_back': (385, 551, 97, 317),
        'pachy': (582, 689, 10, 318),
        'others': (694, 811, 11, 317),
        'white_regions': {
            'cornea_front': [(45, 63, 0, 5)],
            'cornea_back': [(45, 63, 0, 5), (145, 155, 142, 151)]
        }
    },
    (840, 1200): {
        'pat_info': (45, 186, 10, 326),
        'cornea_front': (208, 375, 97, 324),
        'cornea_back': (395, 562, 97, 324),
        'pachy': (595, 705, 10, 325),
        'others': (711, 829, 11, 327),
        'white_regions': {
            'cornea_front': [(46, 64, 0, 7)],
            'cornea_back': [(46, 64, 0, 7), (148, 158, 148, 158)]
        }
    },
    (894, 1600): {
        'pat_info': (47, 198, 11, 344),
        'cornea_front': (218, 393, 103, 344),
        'cornea_back': (415, 592, 103, 344),
        'pachy': (632, 745, 11, 345),
        'others': (755, 880, 11, 347),
        'white_regions': {
            'cornea_front': [(50, 67, 0, 8)],
            'cornea_back': [(50, 67, 0, 8), (158, 169, 157, 168)]
        }
    }
}

REGION_COORDS_PENTACAM = {
    (740, 1200): {
        'pat_info': (43, 157, 6, 258),
        'cornea_front': (171, 311, 80, 257),
        'cornea_back': (319, 454, 80, 257),
        'pachy': (474, 564, 6, 258),
        'others': (565, 662, 5, 257),
        'white_regions': {
            'cornea_front': [(47, 81, 0, 5)],
            'cornea_back': [(47, 81, 0, 5), (119, 128, 117, 125)],
            'pachy': [(6, 85, 152, 161)]
        }
    },
    (838, 1200): {
        'pat_info': (44, 187, 8, 326),
        'cornea_front': (202, 377, 94, 326),
        'cornea_back': (391, 565, 94, 327),
        'pachy': (595, 705, 8, 325),
        'others': (705, 806, 9, 326),
        'white_regions': {
            'cornea_front': [(48, 77, 0, 7)],
            'cornea_back': [(48, 77, 0, 7), (149, 163, 149, 159)]
        }
    },
    (858, 1200): {
        'pat_info': (146, 274, 9, 293),
        'cornea_front': (291, 449, 86, 294),
        'cornea_back': (457, 613, 84, 294),
        'pachy': (638, 739, 9, 294),
        'others': (742, 830, 7, 294),
        'white_regions': {
            'cornea_front': [(47, 78, 0, 8)],
            'cornea_back': [(47, 78, 0, 8), (134, 144, 134, 143)]
        }
    },
    (904, 1200): {
        'pat_info': (111, 254, 9, 327),
        'cornea_front': (270, 442, 96, 326),
        'cornea_back': (456, 627, 95, 326),
        'pachy': (661, 770, 9, 325),
        'others': (770, 872, 11, 326),
        'white_regions': {
            'cornea_front': [(48, 77, 0, 7)],
            'cornea_back': [(48, 77, 0, 7), (151, 162, 148, 157)]
        }
    },
    (910, 1200): {
        'pat_info': (43, 187, 8, 326),
        'cornea_front': (201, 382, 91, 326),
        'cornea_back': (388, 568, 92, 326),
        'pachy': (594, 704, 8, 325),
        'others': (704, 830, 9, 325),
        'white_regions': {
            'cornea_front': [(47, 78, 0, 13)],
            'cornea_back': [(47, 78, 0, 13), (153, 164, 151, 161)]
        }
    },
    (940, 1200): {
        'pat_info': (147, 290, 8, 327),
        'cornea_front': (306, 477, 97, 326),
        'cornea_back': (493, 667, 97, 326),
        'pachy': (697, 804, 8, 325),
        'others': (809, 906, 10, 327),
        'white_regions': {
            'cornea_front': [(47, 81, 0, 5)],
            'cornea_back': [(47, 81, 0, 5), (150, 160, 145, 155)]
        }
    }
}

# Values for specific Pentacam 910_1200 case described in functions.py (region_coords2)
REGION_COORDS_PENTACAM_910_SPECIAL = {
    (910, 1200): {
        'pat_info': (116, 259, 9, 327),
        'cornea_front': (273, 448, 96, 326),
        'cornea_back': (462, 637, 95, 326),
        'pachy': (665, 777, 9, 326),
        'others': (778, 878, 10, 327),
        'white_regions': {
            'cornea_front': [(51, 79, 0, 8)],
            'cornea_back': [(51, 79, 0, 8), (152, 165, 148, 158)]
        }
    }
}


def pad_image_to_min_size(image, min_size=50):
    """
    Pads an image to ensure it meets minimum size requirements.
    """
    if image is None or image.size == 0:
        return image
        
    height, width = image.shape[:2]
    
    # Calculate padding needed
    pad_height = max(0, min_size - height)
    pad_width = max(0, min_size - width)
    
    if pad_height > 0 or pad_width > 0:
        # Create white padding
        if len(image.shape) == 3:
            padded_image = np.ones((height + pad_height, width + pad_width, 3), dtype=np.uint8) * 255
        else:
            padded_image = np.ones((height + pad_height, width + pad_width), dtype=np.uint8) * 255
            
        # Place original image in the center
        padded_image[:height, :width] = image
        return padded_image
    return image


    # Concatenate
    concatenated = cv2.vconcat(result)
    return concatenated


def get_standard_width(regions):
    """Returns the maximum width from a list of images."""
    if not regions:
        return 0
    return max(img.shape[1] for img in regions if img is not None)


def pad_to_width(img, target_width):
    """Pads an image to a target width with white background."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    
    channels = 3 if len(img.shape) == 3 else 1
    if channels == 3:
        padded = np.ones((h, target_width, 3), dtype=np.uint8) * 255
    else:
        padded = np.ones((h, target_width), dtype=np.uint8) * 255
        
    padded[:h, :w] = img
    return padded


def detect_image_type(image, image_path=""):
    """
    Detects the type of map (Pentacam/Oculyzer) and dimensions.
    Returns (map_type, dimensions, is_special_case)
    """
    height, width = image.shape[:2]
    
    # Find matching dimension
    dimension_key = f"{height}_{width}"
    dimensions = (height, width)
    
    # Fuzzy match dimension if exact match not found
    if dimension_key not in used_img_dims:
        for dim in used_img_dims:
            h, w = IMG_SIZE_TO_DIMENSIONS[dim]
            if abs(h - height) < 50 and abs(w - width) < 50:
                dimension_key = dim
                dimensions = (h, w) # Use standard dims? better to use actual if possible or standard keys
                break
    
    # Use standard dimension tuple for lookup
    std_dimensions = IMG_SIZE_TO_DIMENSIONS.get(dimension_key, (height, width))
    
    # Extract header to determine map type
    # Using coords from constants.py
    header_coords = image_header_coord_dict.get(dimension_key, [2, 40, 2, 948])
    try:
        header_region = image[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
        # Use Tesseract or simple string check if possible, but we might not have OCR here just for detection.
        # Alternatively, assume Pentacam vs Oculyzer based on available coords?
        # Actually user said "crop the left side... i have all the dimentions... stored in constants.py"
        # Let's assume standard detection logic if OCR isn't available
        pass
    except:
        pass

    # For now, default to Oculyzer if in its dict, else Pentacam, or check header text if we had OCR.
    # Since we are just cropping, maybe we can try both or user tells us?
    # The previous code distinguished them.
    # Let's try to detect based on which dictionary has the key, prioritizing Oculyzer?
    # Actually, some dimensions overlap (e.g. 758_1200 is in both dicts in functions.py logic? No wait.)
    # In functions.py, Oculyzer has 758, 820, 840, 894.
    # Pentacam has 740, 838, 858, 904, 910, 940.
    # Overlap? 758, 820, 840, 894 seem unique to Oculyzer in that file.
    # 740, 838, 858, 904, 910, 940 seem unique to Pentacam.
    # IMPORTANT: There might be overlap in reality, but based on the provided functions.py, they are disjoint sets.
    
    is_pentacam = std_dimensions in REGION_COORDS_PENTACAM
    is_oculyzer = std_dimensions in REGION_COORDS_OCULYZER
    
    map_type = 'UNKNOWN'
    if is_pentacam:
        map_type = 'PENTACAM'
    elif is_oculyzer:
        map_type = 'OCULYZER'
        
    return map_type, std_dimensions


def crop_left_panel(image_path, output_dir=None):
    """
    Crops the left panel of a Pentacam/Oculyzer map into 5 individual parts.
    
    Args:
        image_path (str): Path to image.
        output_dir (str): Optional directory to save individual parts.
        
    Returns:
        dict: A dictionary of {region_name: image_numpy_array}
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    map_type, dimensions = detect_image_type(image, image_path)
    
    print(f"Processing {os.path.basename(image_path)}")
    print(f"Detected Type: {map_type}, Dimensions: {dimensions}")
    
    coords_set = None
    
    if map_type == 'PENTACAM':
        coords_set = REGION_COORDS_PENTACAM.get(dimensions)
    elif map_type == 'OCULYZER':
        coords_set = REGION_COORDS_OCULYZER.get(dimensions)
    
    if not coords_set:
        print(f"Error: No crop coordinates found for dimensions {dimensions} and type {map_type}")
        return None
        
    # Crop the 5 regions
    regions = {}
    region_names = ['pat_info', 'cornea_front', 'cornea_back', 'pachy', 'others']
    
    for name in region_names:
        if name not in coords_set:
            continue
            
        y1, y2, x1, x2 = coords_set[name]
        
        # Ensure coords are within image bounds
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        
        crop = image[y1:y2, x1:x2].copy()
        
        # Apply white-out if defined
        if 'white_regions' in coords_set and name in coords_set['white_regions']:
            for wy1, wy2, wx1, wx2 in coords_set['white_regions'][name]:
                # White regions are relative to the crop
                h, w = crop.shape[:2]
                wy1, wy2 = max(0, wy1), min(h, wy2)
                wx1, wx2 = max(0, wx1), min(w, wx2)
                
                crop[wy1:wy2, wx1:wx2] = 255
                
        regions[name] = crop

    # Normalize widths to the maximum width among regions for visual consistency
    max_w = get_standard_width(list(regions.values()))
    for name in regions:
        regions[name] = pad_to_width(regions[name], max_w)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for name, img in regions.items():
            save_path = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Saved {name} to {save_path}")
        
    return regions

if __name__ == "__main__":
    # Specify your image path here
    image_path = r"C:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\Suture_tension_dataset\P769828\2304773_0.814234001761211805.jpg"
    
    # Specify output directory
    output_dir = r"C:\Users\shivam.prajapati\Desktop\4MAP_OCR\crops"
    
    crop_left_panel(image_path, output_dir)