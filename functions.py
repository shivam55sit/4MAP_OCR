import cv2
import numpy as np
import pandas as pd
import os
import re

from tqdm import tqdm
import concurrent.futures

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from google.cloud import vision
import io
import easyocr
from consts import *

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

SUBSCRIPTION_KEY = os.getenv("VISION_KEY")
ENDPOINT = os.getenv("VISION_ENDPOINT")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/home/optigene-dev-vm1-admin/keratoconus/kc_project_phase3/vision_api.json'

os.chdir('/home/optigene-dev-vm1-admin/keratoconus')
reader = easyocr.Reader(['en'])



####### Common functions #########

def replace_path_in_dataframe(df):
    """
    Replace the specified path strings in all columns of a dataframe using path_replace_dict
    """
    path_replace_dict = {
    'D:/OneDrive - Optigene Private Limited/keratoconus/kc_project_phase3/DE/v3/Keratoconus-Images-Phase-3':'/mnt/blobstorage/Keratoconus-Images-Phase-3',
    'D:/OneDrive - Optigene Private Limited/keratoconus/kc_project_phase3/DE/v3/kc_phase3_new_inv_images':'/mnt/blobstorage/kc_phase3_new_inv_images',
    'D:/OneDrive - Optigene Private Limited/keratoconus/kc_project_phase3/DE/v3/missed_images_kc_phase3':'/mnt/blobstorage/missed_images_kc_phase3',
    'D:/OneDrive - Optigene Private Limited/keratoconus/kc_project_phase3/DE/v3/Missed_138_MRNs':'/mnt/blobstorage/Missed_138_MRNs',
 
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Oculyzer':'/mnt/blobstorage/Oculyzer',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Oculyzer_2':'/mnt/blobstorage/Oculyzer_2',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Oculyzer_3_jpg':'/mnt/blobstorage/Oculyzer_3_jpg',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Pentacam/Keratoconus_with_values_without_markings (1-900)':'/mnt/blobstorage/Pentacam/Keratoconus_with_values_without_markings (1-900)',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Pentacam/Keratoconus_with_values (901-1275)':'/mnt/blobstorage/Pentacam/Keratoconus_with_values (901-1275)',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Pentacam/Keratoconus_with_only_Elevation_values (1276-1953)':'/mnt/blobstorage/Pentacam/Keratoconus_with_only_Elevation_values (1276-1953)',
    'D:/OneDrive - Optigene Private Limited/keratoconus/data/Keratoconus_060224':'/mnt/blobstorage/Keratoconus_060224',
    'D:/OneDrive - Optigene Private Limited/E_Drive/CombinedFolder':'/mnt/blobstorage/CombinedFolder'
    }
	
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':  # Only process string columns
            # First handle the path replacements only on non-null values
            mask = df_copy[col].notna()
            if mask.any():
                for old_path, new_path in path_replace_dict.items():
                    df_copy.loc[mask, col] = df_copy.loc[mask, col].str.replace(old_path, new_path, regex=False)
    return df_copy


def compare(x,y):
    """
    Compares two sets and prints their differences and intersections.
    
    This function compares two iterables by converting them to sets and calculating
    the differences and intersections between them. It prints the number of elements
    that are unique to each set and the number of elements they share.
    
    Args:
        x (iterable): First iterable to compare
        y (iterable): Second iterable to compare
        
    Prints:
        - Number of elements in x that are not in y
        - Number of elements in y that are not in x
        - Number of elements common to both x and y (printed twice for verification)
    """
    print('Difference 1 vs 2: ', len(set(x).difference(set(y))))
    print('Difference 2 vs 1: ', len(set(y).difference(set(x))))

    print('Intersection 1 vs 2: ', len(set(x).intersection(set(y))))
    print('Intersection 2 vs 1: ', len(set(y).intersection(set(x))))

def compare_values(x,y):
    print('Difference 1 vs 2: ', set(x).difference(set(y)))
    print('Difference 2 vs 1: ', set(y).difference(set(x)))

    print('Intersection 1 vs 2: ', set(x).intersection(set(y)))
    print('Intersection 2 vs 1: ', set(y).intersection(set(x)))

def get_image_name(x):
    x = x.replace('\\', '/')
    return x.split('/')[-1]


################## OCR Common Functions ##################################

def op_to_df(ip_img):
    """
    Convert easy OCR output from an image to a structured pandas DataFrame.
    
    This function takes an image as input, performs OCR using EasyOCR, and converts
    the output into a DataFrame with columns for location, value, and confidence.
    
    Args:
        ip_img (numpy.ndarray): Input image array to perform OCR on
        
    Returns:
        pandas.DataFrame: DataFrame containing OCR results with columns:
            - Location: Bounding box coordinates of detected text
            - Value: Extracted text content
            - Confidence: Confidence score of the OCR detection
    """
    output = reader.readtext(ip_img)
    df = pd.DataFrame(columns = ['Location', 'Value', 'Confidence'])
    rows = [
        {'Location': output[i][0], 'Value': output[i][1], 'Confidence': output[i][2]}
        for i in range(len(output))
        ]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

def detect_text(path):
    """
    Detects and extracts text from an image using Google Cloud Vision API.
    
    This function takes an image file path, sends it to Google Cloud Vision API for text detection,
    and returns both the detected text and their corresponding vertex coordinates.
    
    Args:
        path (str): Path to the image file to be processed
        
    Returns:
        list: A list containing two elements:
            - List of detected text strings (excluding the first element which is typically the full text)
            - List of vertex coordinates for each text element in format [(x1,y1), (x2,y2), ...]
            
    Raises:
        Exception: If the Google Cloud Vision API returns an error message
    """
    client = vision.ImageAnnotatorClient()
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    lst_text = []
    ver_lst = []
    for text in texts:
        lst_text.append(text.description)
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])
        ver_lst.append(vertices)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        
    ret_list = list()
    for df in [lst_text[1:], ver_lst[1:]]:
        ret_list.append(df)

    return(ret_list)

def sort_by_cor(ret_data):
    """
    Sorts detected text elements based on their spatial coordinates.
    
    This function takes the output from text detection and sorts the text elements
    first by vertical position (y-coordinate), then by horizontal position (x-coordinate)
    within each line. Text elements within 5 pixels vertically are considered to be
    on the same line.
    
    Args:
        ret_data (list): List containing detected text and vertex coordinates
            [0] - List of text strings
            [1] - List of vertex coordinates
            
    Returns:
        list: Sorted list of text strings based on their spatial arrangement
    """
    pat_info_lst = ret_data[0]
    pat_info_ver = ret_data[1]
    
    y_cor = [x[0] for x in pat_info_ver]
    y_cor = [x.replace(")", "") for x in [x.replace("(", "") for x in y_cor]]
    y_cor = np.array([x.strip().split(',') for x in y_cor], dtype=int)
    y_cor.astype(int)
    y_cor = y_cor.tolist()    
    y = sorted(y_cor, key=lambda x : x[1])
    
    x_list = []
    y_list = []
    reset_index = []
    for i in range(len(y)+1):
        if i == len(y):
            re_in = list(np.argsort(x_list))
            re_in = [(i-len(x_list)+x) for x in re_in]
            [reset_index.append(x) for x in re_in]
        else:
            if i == 0:
                y_list.append(y[i][1])
                x_list.append(y[i][0])
            else:
                y_diff = y_list[len(y_list)-1] - y[i][1]
                if y_diff >= -5 and y_diff <= 5:
                    y_list.append(y[i][1])
                    x_list.append(y[i][0])
                else:
                    re_in = list(np.argsort(x_list))
                    re_in = [(i-len(x_list)+x) for x in re_in]
                    [reset_index.append(x) for x in re_in]
                    x_list = []
                    y_list = []

                    x_list.append(y[i][0])
                    y_list.append(y[i][1])
    y1 = np.array(y)
    sort_ind = np.array(reset_index)
    y1 = y1[sort_ind].tolist()
    
    new_index = []
    for i in range(len(y1)):
        new_index.append(y_cor.index(y1[i]))
    
    pat_info_lst = np.array(pat_info_lst)
    sort_ind = np.array(new_index)
    pat_info_lst = pat_info_lst[sort_ind].tolist()
    
    return pat_info_lst

def extract_text(image_path):
    """
    Extracts text from an image using Tesseract OCR.
    
    This function reads an image file and uses Tesseract OCR engine to
    extract text content from it. The image is processed as is, without
    any preprocessing.
    
    Args:
        image_path (str): Path to the image file to be processed
        
    Returns:
        str: Extracted text from the image, with leading and trailing whitespace removed
    """
    image = cv2.imread(image_path, 1)
    text = pytesseract.image_to_string(image)
    return text.strip()

def convert_coordinates(coords):
    """
    Converts coordinate dictionaries to formatted coordinate strings.
    
    This function takes a list of coordinate dictionaries and converts them
    to a list of formatted strings in the format "(x,y)".
    
    Args:
        coords (list): List of dictionaries containing 'x' and 'y' coordinates
        
    Returns:
        list: List of formatted coordinate strings in "(x,y)" format
    """
    return [f"({coord['x']},{coord['y']})" for coord in coords]

def detect_text_azure(image_path):
    """
    Detects and extracts text from an image using Azure Computer Vision API.
    
    This function processes an image using Azure's Computer Vision service to extract
    text at both the line and word level, along with their bounding coordinates.
    
    Args:
        image_path (str): Path to the image file to be processed
        
    Returns:
        tuple: Two elements:
            - line_items (list): [line_text, line_coords] for detected lines
            - word_items (list): [word_text, word_coords] for individual words
        where text is the detected text and coords are the bounding polygon coordinates
    """
    client = ImageAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(SUBSCRIPTION_KEY)
    )

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    result = client.analyze(image_data=image_data,
                            visual_features=[VisualFeatures.READ]
                            )
    line_text = []
    line_coords = []

    word_text = []
    word_coords = []

    if result.read is not None:
        for line in result.read.blocks[0].lines:
            line_text.append(line.text)
            line_coords.append(convert_coordinates(line.bounding_polygon))
            for word in line.words:
                word_text.append(word.text)
                word_coords.append(convert_coordinates(word.bounding_polygon))
    line_items = [line_text, line_coords]
    word_items = [word_text, word_coords]

    return line_items, word_items

def concat_images_vertical_with_spacing(images, spacing=10):
    """
    Concatenates multiple images vertically with uniform spacing between them.
    
    This function takes a list of images and combines them vertically, adding
    white spacing between each image. All images are resized to have the same
    width while maintaining their aspect ratios.
    
    Args:
        images (list): List of numpy arrays representing images to be concatenated
        spacing (int, optional): Number of pixels of white space to add between images.
            Defaults to 10.
            
    Returns:
        numpy.ndarray: Single concatenated image with all input images stacked vertically
            with uniform spacing
    """
    # Ensure all images have the same width
    min_width = min(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (min_width, int(img.shape[0] * (min_width/img.shape[1])))) 
                     for img in images]
    
    # Create spacing image
    spacing_img = np.ones((spacing, min_width, 3), dtype=np.uint8) * 255  # White spacing
    
    # Add spacing between images
    result = []
    for i, img in enumerate(resized_images):
        result.append(img)
        if i < len(resized_images) - 1:  # Don't add spacing after the last image
            result.append(spacing_img)
    
    # Concatenate all images and spacing
    concatenated = cv2.vconcat(result)
    return concatenated

def pad_image_to_min_size(image, min_size=50):
    """
    Pads an image to ensure it meets minimum size requirements.
    
    This function adds white padding to an image if either its height or width
    is less than the specified minimum size. The original image is centered
    in the padded result.
    
    Args:
        image (numpy.ndarray): Input image to be padded
        min_size (int, optional): Minimum size requirement for both dimensions.
            Defaults to 50.
            
    Returns:
        numpy.ndarray: Padded image that meets minimum size requirements
    """
    height, width = image.shape[:2]
    
    # Calculate padding needed
    pad_height = max(0, min_size - height)
    pad_width = max(0, min_size - width)
    
    if pad_height > 0 or pad_width > 0:
        # Create white padding
        padded_image = np.ones((height + pad_height, width + pad_width, 3), dtype=np.uint8) * 255
        # Place original image in the center
        padded_image[:height, :width] = image
        return padded_image
    return image



def concat_images_vertical_with_spacing_for_metrics(images, spacing=10):
   
    # Find the maximum width among all images
    max_width = max(img.shape[1] for img in images)
    
    # Create padded versions of images to match max_width
    padded_images = []
    for img in images:
        if img.shape[1] < max_width:
            # Calculate padding needed on each side
            pad_width = max_width - img.shape[1]
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            
            # Create white padding
            padded_img = np.ones((img.shape[0], max_width, 3), dtype=np.uint8) * 255
            # Place original image in the center
            padded_img[:, left_pad:left_pad + img.shape[1]] = img
            padded_images.append(padded_img)
        else:
            padded_images.append(img)
    
    # Create spacing image with max_width
    spacing_img = np.ones((spacing, max_width, 3), dtype=np.uint8) * 255  # White spacing
    
    # Add spacing between images
    result = []
    for i, img in enumerate(padded_images):
        result.append(img)
        if i < len(padded_images) - 1:  # Don't add spacing after the last image
            result.append(spacing_img)
    
    # Concatenate all images and spacing
    concatenated = cv2.vconcat(result)
    return concatenated

def get_region_indices(img_concat_data, original_regions, spacing=5):
    """
    Identifies which text coordinates belong to which original region in the concatenated image.
    
    Args:
        img_concat_data (list): Output from detect_text containing [text_list, vertex_coordinates]
        original_regions (dict): Dictionary of original image regions and their arrays
        spacing (int): Spacing used between images in concatenation
        
    Returns:
        dict: Dictionary mapping each region to the indices of its text elements
    """
    # Get the vertex coordinates
    vertices = img_concat_data[1]
    
    # Convert vertex strings to actual coordinates
    y_coordinates = []
    for vertex_set in vertices:
        # Get the y-coordinate from the first vertex (top-left corner)
        y_coord = int(vertex_set[0].split(',')[1].strip(')'))
        y_coordinates.append(y_coord)
    
    # Calculate the boundaries for each region
    boundaries = []
    current_height = 0
    
    for i, (name, region) in enumerate(original_regions.items()):
        region_height = region.shape[0]
        
        # Add the region boundaries
        boundaries.append({
            'name': name,
            'start': current_height,
            'end': current_height + region_height
        })
        
        # Update current_height for next region
        current_height += region_height
        
        # Add spacing after each region (except the last one)
        if i < len(original_regions) - 1:
            current_height += spacing
    
    # Map each text element to its region
    region_indices = {name: [] for name in original_regions.keys()}
    
    for idx, y_coord in enumerate(y_coordinates):
        # Find which region this y-coordinate belongs to
        for boundary in boundaries:
            if boundary['start'] <= y_coord <= boundary['end']:
                region_indices[boundary['name']].append(idx)
                break
    
    return region_indices


########## TEC.2 ############################

def text_extraction_oculyzer(img_files, img_size):
    """
    Unified function for text extraction from Oculyzer images.
    
    This function processes a batch of Oculyzer images to extract various metrics including
    patient information, corneal measurements, and other diagnostic data. It handles images
    of different sizes and formats the extracted data into a structured DataFrame.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted metrics with columns:
            - img_file: Path to the source image
            - name: Patient name
            - MRNumber: Medical record number
            - dob: Date of birth
            - eye: Left/Right eye indicator
            - exam_date: Examination date
            - time: Examination time
            - Various corneal measurements (Rf, k1, Rs, k2, etc.)
            - Various location measurements (pupil center, pachy apex, etc.)
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    

    # Pre-compile regex patterns
    non_numeric_pattern = re.compile("[^0-9-.]")
    
    # Common column names for all image sizes
    columns = ['img_file', 'name', 'MRNumber', 'dob','eye','exam_date','time',
               'Rf_cf', 'k1_cf','Rs_cf', 'k2_cf', 'Rm_cf', 'Km_cf', 
               'Axis_cf', 'Astig_cf', 'Rper_cf', 'Rmin_cf',
               'Rf_cb', 'k1_cb','Rs_cb', 'k2_cb', 'Rm_cb', 'Km_cb', 
               'Axis_cb', 'Astig_cb', 'Rper_cb', 'Rmin_cb',
               'pupil_center', 'pc_x', 'pc_y', 'pachy_apex', 'pa_x', 'pa_y',
               'thinnest_loc', 'tl_x', 'tl_y', 'kmax_front', 'kmax_x', 'kmax_y',
               'cornea_volume', 'kpd', 'chamber_vol', 'angle', 'ac_dept_int', 'pupil_dia']

     # Image region coordinates for different sizes
    region_coords = {
        (758, 1200): {
            'pat_info': (43, 173, 10, 295),
            'cornea_front': (188, 353, 88, 295),
            'cornea_back': (357, 521, 87, 295),
            'pachy': (541, 639, 120, 295),
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
            'pachy': (582, 689, 125, 318),
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
            'pachy': (595, 705, 132, 325),
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
            'pachy': (632, 745, 140, 345),
            'others': (755, 880, 11, 347),
            'white_regions': {
                'cornea_front': [(50, 67, 0, 8)],
                'cornea_back': [(50, 67, 0, 8), (158, 169, 157, 168)]
            }
        }
    }


    if img_size not in region_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    coords = region_coords[img_size]
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)

    # Common character lists for filtering
    char_list2 = frozenset(["",".",':','|','O',')','::','o','0','μm','(','D','mm³','mm²','*','mm','[',']','+','flt','<','flat', '>>'])
    pat_info_drop = frozenset(['Last','Name','First','Name'])

    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            pat_metrics = [img_file]

            # Extract regions based on coordinates
            regions = {
                'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                              coords['pat_info'][2]:coords['pat_info'][3]],
                'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                  coords['cornea_front'][2]:coords['cornea_front'][3]],
                'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                 coords['cornea_back'][2]:coords['cornea_back'][3]],
                'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                           coords['pachy'][2]:coords['pachy'][3]],
                'others': img[coords['others'][0]:coords['others'][1], 
                            coords['others'][2]:coords['others'][3]]
            }

            # Apply white regions
            for region, white_areas in coords['white_regions'].items():
                for y1, y2, x1, x2 in white_areas:
                    regions[region][y1:y2, x1:x2] = 255

            # Save and process regions
            paths = {name: f'{temp_dir}/{name}_{os.path.basename(img_file)}' 
                    for name in regions}
            
            for name, region in regions.items():
                cv2.imwrite(paths[name], region)

            data = {name: sort_by_cor(detect_text(path)) 
                   for name, path in paths.items()}

            # Clean up temporary files
            for path in paths.values():
                try:
                    os.remove(path)
                except:
                    pass

            # Process patient info
            pat_info_lst = [x.replace("|", "") for x in data['pat_info'] 
                           if x not in char_list2]
            pat_info_lst_new = [x for x in pat_info_lst if x not in pat_info_drop]
            
            try:
                id_index = pat_info_lst_new.index('ID')
                pat_name = ' '.join(pat_info_lst_new[0:id_index])

                if pat_info_lst_new[id_index+2] == '-':
                    mrn = ''.join(str(x) for x in pat_info_lst_new[(id_index+1):(id_index+4)])
                else:
                    mrn = str(pat_info_lst_new[id_index + 1])
                    
                pat_info_lst_fi = [
                    pat_name,
                    mrn,
                    pat_info_lst_new[pat_info_lst_new.index('Birth') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Eye') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Exam') + 2],
                    pat_info_lst_new[pat_info_lst_new.index('Time') + 1]
                ]                
                # Handle special eye cases
                if len([x for x in ['Left', 'Right'] if x not in pat_info_lst_fi]) == 2:
                    ind = [pat_info_lst_new.index(x) for x in ['Left', 'Right'] 
                          if x in pat_info_lst_new]
                    if ind:
                        pat_info_lst_fi[3] = pat_info_lst_new[ind[0]]
                
                pat_metrics.extend(pat_info_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            # Process cornea regions
            for region in ['cornea_front', 'cornea_back']:
                try:
                    region_lst = [x.replace("|", "") for x in data[region] 
                                if x not in char_list2]
                    
                    check = next((x for x in ["Rt", "Rf", "R"] if x in region_lst), "R")
                    res = [ele for ele in region_lst if "min" in ele]
                    
                    region_lst_fi = [
                        region_lst[region_lst.index(check) + 1],
                        region_lst[region_lst.index('K1') + 1],
                        region_lst[region_lst.index('Rs') + 1],
                        region_lst[region_lst.index('K2') + 1],
                        region_lst[region_lst.index('Rm') + 1],
                        region_lst[region_lst.index('Km') + 1],
                        region_lst[region_lst.index('Axis') + 1],
                        region_lst[region_lst.index('Astig') + 1],
                        region_lst[region_lst.index('Rper') + 1],
                        region_lst[region_lst.index(res[0]) + 1]
                    ]
                    pat_metrics.extend(region_lst_fi)
                except Exception as e:
                    pat_metrics.extend([''] * 10)

            # Process pachy data
            try:
                pachy_lst = [non_numeric_pattern.sub("", x) for x in data['pachy']]
                pachy_lst = [x for x in pachy_lst if x not in char_list2]
                
                if len(pachy_lst) <= 9:
                    pachy_lst = ['nan', 'nan', 'nan'] + [str(x) for x in pachy_lst]
                
                if len(pachy_lst) < 12:
                    for i, loc in enumerate([0, 3, 6, 9]):
                        if float(pachy_lst[loc]) < 10:
                            pachy_lst.insert(loc, 'nan')
                
                pat_metrics.extend([x.replace("|", "") for x in pachy_lst[:12]])
            except Exception as e:
                pat_metrics.extend([''] * 12)

            # Process others data
            try:
                others_lst = [x.replace("|", "") for x in data['others'] 
                            if x not in char_list2]
                
                check = "Int" if "Int" in others_lst else "Ext"
                others_lst_fi = [
                    others_lst[others_lst.index('Volume') + 1],
                    others_lst[others_lst.index('KPD') + 1],
                    others_lst[others_lst.index('Chamber') + 2],
                    others_lst[others_lst.index('Angle') + 1],
                    others_lst[others_lst.index(check) + 1],
                    others_lst[others_lst.index('Dia') + 1]
                ]
                pat_metrics.extend(others_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df


################## TEC.3 ##################################

def text_extraction_patient_info(img_files, img_size):
    """
    Extracts patient information from PENTACAM/Oculyzer images.
    
    This function processes a batch of images to extract patient demographic and
    examination information from specific regions of the images. It supports multiple
    image sizes and formats, with appropriate coordinate mapping for each format.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        
    Returns:
        pandas.DataFrame: DataFrame containing patient information with columns:
            - img_file: Path to the source image
            - name: Patient name
            - MRNumber: Medical record number
            - dob: Date of birth
            - eye: Left/Right eye indicator
            - exam_date: Examination date
            - time: Examination time
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)

    columns = ['img_file', 'name', 'MRNumber', 'dob', 'eye', 'exam_date', 'time']
    

     # Region coordinates for different sizes
    region_coords = {
        (740, 1200): {
            'pat_info': (45, 153, 9, 250)
        },
        (838, 1200): {
            'pat_info': (45, 163, 11, 325)
        },
        (858, 1200): {
            'pat_info': (146, 274, 9, 293)
        },
        (904, 1200): {
            'pat_info': (110, 254, 9, 325)
        },
        (910, 1200): {
            'header': (2, 40, 2, 948),
            'pat_info': (45, 185, 9, 325)
        },
        (940, 1200): {
            'pat_info': (146, 289, 10, 327)
        },

        (758, 1200): {
            'pat_info': (43, 173, 10, 295)
        },
        (820, 1200): {
            'pat_info': (45, 186, 10, 321)
        },
        (840, 1200): {
            'pat_info': (45, 186, 10, 326)
        },
        (894, 1600): {
            'pat_info': (47, 198, 11, 344)
        }
    }


    if img_size not in region_coords:
        raise ValueError(f"Unsupported PENTACAM image size: {img_size}")

    # Common character lists for filtering
    char_list2 = frozenset(["",".",':','|','O',')','::','o','0','μm','(','D','mm³','mm²','*','mm','[',']','+','flt','<','flat'])
    pat_info_drop = frozenset(['Last','Name','First','Name'])

    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file,1)
            coords = {k: v for k, v in region_coords[img_size].items()}

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    coords['header'] = (coords['header'][0] + 70, coords['header'][1] + 72, 
                                      coords['header'][2], coords['header'][3])
                    coords['pat_info'] = (coords['pat_info'][0] + 70, coords['pat_info'][1] + 72, 
                                        coords['pat_info'][2], coords['pat_info'][3])
            # print(coords)
                        
            pat_metrics = [img_file]

            # Extract patient info region
            pat_info = img[coords['pat_info'][0]:coords['pat_info'][1],
                         coords['pat_info'][2]:coords['pat_info'][3]]

            # Save temporary file
            pat_info_path = f'{temp_dir}/pat_info_{os.path.basename(img_file)}'
            cv2.imwrite(pat_info_path, pat_info)

            # Process text data
            pat_info_data = detect_text(pat_info_path)
            pat_info_lst = sort_by_cor(pat_info_data)

            # Clean up temporary file
            try:
                os.remove(pat_info_path)
            except:
                pass

            # Process patient info
            pat_info_lst = [x.replace("|", "") for x in pat_info_lst 
                           if x not in char_list2]
            pat_info_lst_new = [x for x in pat_info_lst 
                              if x not in pat_info_drop]
            
            try:
                id_index = pat_info_lst_new.index('ID')
                pat_name = ' '.join(pat_info_lst_new[0:id_index])
                if pat_info_lst_new[id_index+2] == '-':
                    mrn = ''.join(str(x) for x in pat_info_lst_new[(id_index+1):(id_index+4)])
                else:
                    mrn = str(pat_info_lst_new[id_index + 1])

                pat_info_lst_fi = [
                    pat_name,
                    mrn,
                    pat_info_lst_new[pat_info_lst_new.index('Birth') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Eye') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Exam') + 2],
                    pat_info_lst_new[pat_info_lst_new.index('Time') + 1]
                ]
                
                # Handle special eye cases
                if len([x for x in ['Left', 'Right'] if x not in pat_info_lst_fi]) == 2:
                    ind = [pat_info_lst_new.index(x) for x in ['Left', 'Right'] 
                          if x in pat_info_lst_new]
                    if ind:
                        pat_info_lst_fi[3] = pat_info_lst_new[ind[0]]
                
                pat_metrics.extend(pat_info_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df


######## TEC.5 ########################################

def text_extraction_map_header(img_files, img_size, img_header):
    """
    Extracts map header text from diagnostic images using Tesseract OCR.
    
    This function processes images to extract text from four header regions in diagnostic maps.
    It handles different image sizes and formats, and can process both PENTACAM and OCULYZER
    images. The function uses Tesseract OCR for text extraction and includes special handling
    for different image layouts.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        img_header (str): Type of image header ('OCULUS PENTACAM 4 Maps Selectable',
                         'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable', etc.)
        
    Returns:
        pandas.DataFrame: DataFrame containing header information with columns:
            - img_file: Path to the source image
            - header1: Text from first header region
            - header2: Text from second header region
            - header3: Text from third header region
            - header4: Text from fourth header region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file','header1', 'header2', 'header3', 'header4']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)

            if img_header in ['OCULUS PENTACAM 4 Maps Selectable', 'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable']:
                h_coords = {k: [v[0]-1, v[1]+1, v[2]-1, v[3]+1] for k, v in selectable_map_header_coords[img_size].items()}
            else:
                h_coords = {k: [v[0]-1, v[1]+1, v[2]-1, v[3]+1] for k, v in refractive_map_header_coords[img_size].items()}

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    h_coords = {k: [v[0]-1, v[1]+1, v[2]-1, v[3]+1] for k, v in refractive_map_header_coords2[img_size].items()}

            pat_metrics = [img_file]

            head1_coor = h_coords['header1']
            head2_coor = h_coords['header2']
            head3_coor = h_coords['header3']
            head4_coor = h_coords['header4']

            header1 = img[head1_coor[0]:head1_coor[1], head1_coor[2]:head1_coor[3]]
            header2 = img[head2_coor[0]:head2_coor[1], head2_coor[2]:head2_coor[3]]
            header3 = img[head3_coor[0]:head3_coor[1], head3_coor[2]:head3_coor[3]]
            header4 = img[head4_coor[0]:head4_coor[1], head4_coor[2]:head4_coor[3]]

            header1_path = f'{temp_dir}/header1_{os.path.basename(img_file)}'
            header2_path = f'{temp_dir}/header2_{os.path.basename(img_file)}'
            header3_path = f'{temp_dir}/header3_{os.path.basename(img_file)}'
            header4_path = f'{temp_dir}/header4_{os.path.basename(img_file)}'

            cv2.imwrite(header1_path, header1)
            cv2.imwrite(header2_path, header2)
            cv2.imwrite(header3_path, header3)
            cv2.imwrite(header4_path, header4)

            header1_text = extract_text(header1_path)
            header2_text = extract_text(header2_path)
            header3_text = extract_text(header3_path)
            header4_text = extract_text(header4_path)

            # Clean up temporary file
            try:
                os.remove(header1_path)
                os.remove(header2_path)
                os.remove(header3_path)
                os.remove(header4_path)
            except:
                pass

            # print(header1_text, header2_text, header3_text, header4_text)

            pat_metrics.extend([header1_text, header2_text, header3_text, header4_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_map_header_google(img_files, img_size, img_header):
    """
    Extracts map header text from diagnostic images using Google Cloud Vision API.
    
    Similar to text_extraction_map_header(), but uses Google Cloud Vision API for text
    extraction instead of Tesseract OCR. This may provide better accuracy for challenging
    text or low-quality images. The function processes four header regions and includes
    spatial sorting of detected text.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        img_header (str): Type of image header ('OCULUS PENTACAM 4 Maps Selectable',
                         'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable', etc.)
        
    Returns:
        pandas.DataFrame: DataFrame containing header information with columns:
            - img_file: Path to the source image
            - header1: Text from first header region
            - header2: Text from second header region
            - header3: Text from third header region
            - header4: Text from fourth header region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file','header1', 'header2', 'header3', 'header4']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)

            if img_header in ['OCULUS PENTACAM 4 Maps Selectable', 'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable']:
                h_coords = {k: v for k, v in selectable_map_header_coords[img_size].items()}
            else:
                h_coords = {k: v for k, v in refractive_map_header_coords[img_size].items()}

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    h_coords = {k: v for k, v in refractive_map_header_coords2[img_size].items()}

            pat_metrics = [img_file]

            head1_coor = h_coords['header1']
            head2_coor = h_coords['header2']
            head3_coor = h_coords['header3']
            head4_coor = h_coords['header4']

            header1 = img[head1_coor[0]:head1_coor[1], head1_coor[2]:head1_coor[3]]
            header2 = img[head2_coor[0]:head2_coor[1], head2_coor[2]:head2_coor[3]]
            header3 = img[head3_coor[0]:head3_coor[1], head3_coor[2]:head3_coor[3]]
            header4 = img[head4_coor[0]:head4_coor[1], head4_coor[2]:head4_coor[3]]

            header1_path = f'{temp_dir}/header1_{os.path.basename(img_file)}'
            header2_path = f'{temp_dir}/header2_{os.path.basename(img_file)}'
            header3_path = f'{temp_dir}/header3_{os.path.basename(img_file)}'
            header4_path = f'{temp_dir}/header4_{os.path.basename(img_file)}'

            cv2.imwrite(header1_path, header1)
            cv2.imwrite(header2_path, header2)
            cv2.imwrite(header3_path, header3)
            cv2.imwrite(header4_path, header4)


            header1_data = detect_text(header1_path)
            header2_data = detect_text(header2_path)
            header3_data = detect_text(header3_path)
            header4_data = detect_text(header4_path)

            header1_data_lst = sort_by_cor(header1_data)
            header2_data_lst = sort_by_cor(header2_data)
            header3_data_lst = sort_by_cor(header3_data)
            header4_data_lst = sort_by_cor(header4_data)

            header1_text = ' '.join(header1_data_lst)
            header2_text = ' '.join(header2_data_lst)
            header3_text = ' '.join(header3_data_lst)
            header4_text = ' '.join(header4_data_lst)

            # Clean up temporary file
            try:
                os.remove(header1_path)
                os.remove(header2_path)
                os.remove(header3_path)
                os.remove(header4_path)
            except:
                pass

            # print(header1_text, header2_text, header3_text, header4_text)

            pat_metrics.extend([header1_text, header2_text, header3_text, header4_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_map_header_azure(img_files, img_size, img_header):
    """
    Extracts map header text from diagnostic images using Azure Computer Vision API.
    
    Similar to text_extraction_map_header(), but uses Azure's Computer Vision API for text
    extraction. This version includes additional image preprocessing steps like padding and
    may provide better results for certain types of images. The function processes four
    header regions and maintains spatial relationships between detected text elements.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        img_header (str): Type of image header ('OCULUS PENTACAM 4 Maps Selectable',
                         'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable', etc.)
        
    Returns:
        pandas.DataFrame: DataFrame containing header information with columns:
            - img_file: Path to the source image
            - header1: Text from first header region
            - header2: Text from second header region
            - header3: Text from third header region
            - header4: Text from fourth header region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file','header1', 'header2', 'header3', 'header4']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            if img_header in ['OCULUS PENTACAM 4 Maps Selectable', 'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable']:
                h_coords = {k: v for k, v in selectable_map_header_coords[img_size].items()}
            else:
                h_coords = {k: v for k, v in refractive_map_header_coords[img_size].items()}

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    h_coords = {k: v for k, v in refractive_map_header_coords2[img_size].items()}

            pat_metrics = [img_file]

            head1_coor = h_coords['header1']
            head2_coor = h_coords['header2']
            head3_coor = h_coords['header3']
            head4_coor = h_coords['header4']

            header1 = img[head1_coor[0]:head1_coor[1], head1_coor[2]:head1_coor[3]]
            header2 = img[head2_coor[0]:head2_coor[1], head2_coor[2]:head2_coor[3]]
            header3 = img[head3_coor[0]:head3_coor[1], head3_coor[2]:head3_coor[3]]
            header4 = img[head4_coor[0]:head4_coor[1], head4_coor[2]:head4_coor[3]]

            header1 = pad_image_to_min_size(header1)
            header2 = pad_image_to_min_size(header2)
            header3 = pad_image_to_min_size(header3)
            header4 = pad_image_to_min_size(header4)

            # headerall = concat_images_vertical_with_spacing([header1, header2, header3, header4],
            #                                                 spacing=20)

            header1_path = f'{temp_dir}/header1_{os.path.basename(img_file)}'
            header2_path = f'{temp_dir}/header2_{os.path.basename(img_file)}'
            header3_path = f'{temp_dir}/header3_{os.path.basename(img_file)}'
            header4_path = f'{temp_dir}/header4_{os.path.basename(img_file)}'
            # headerall_path = f'{temp_dir}/headerall_{os.path.basename(img_file)}'

            cv2.imwrite(header1_path, header1)
            cv2.imwrite(header2_path, header2)
            cv2.imwrite(header3_path, header3)
            cv2.imwrite(header4_path, header4)
            # cv2.imwrite(headerall_path, headerall)

            # cv2.imshow('header1', header1)
            # cv2.imshow('header2', header2)
            # cv2.imshow('header3', header3)
            # cv2.imshow('header4', header4)
            # cv2.imshow('headerall', headerall)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            _, header1_data = detect_text_azure(header1_path)
            _, header2_data = detect_text_azure(header2_path)
            _, header3_data = detect_text_azure(header3_path)
            _, header4_data = detect_text_azure(header4_path)

            header1_data_lst = sort_by_cor(header1_data)
            header2_data_lst = sort_by_cor(header2_data)
            header3_data_lst = sort_by_cor(header3_data)
            header4_data_lst = sort_by_cor(header4_data)

            header1_text = ' '.join(header1_data_lst)
            header2_text = ' '.join(header2_data_lst)
            header3_text = ' '.join(header3_data_lst)
            header4_text = ' '.join(header4_data_lst)

            # Clean up temporary file
            try:
                os.remove(header1_path)
                os.remove(header2_path)
                os.remove(header3_path)
                os.remove(header4_path)
            except:
                pass


            # headerall_data, _ = detect_text_azure(headerall_path)
            # Get heights of individual images
            # image_heights = [header1.shape[0], header2.shape[0], header3.shape[0], header4.shape[0]]
            # headerall_data_lst = combine_text_by_region(headerall_data, image_heights, spacing=20)

            # header1_text = headerall_data_lst[0]
            # header2_text = headerall_data_lst[1]
            # header3_text = headerall_data_lst[2]
            # header4_text = headerall_data_lst[3]

            # print(header1_text, header2_text, header3_text, header4_text)

            pat_metrics.extend([header1_text, header2_text, header3_text, header4_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df


######## TEC.6 ########################################

def text_extraction_QS(img_files, img_size):
    """
    Extracts QS (Quality Specification) text from diagnostic images using Tesseract OCR.
    
    This function processes images to extract text from QS regions, including both front
    and back measurements and their corresponding values. It uses Tesseract OCR for
    text extraction and handles different image sizes and formats.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        
    Returns:
        pandas.DataFrame: DataFrame containing QS information with columns:
            - img_file: Path to the source image
            - qs_front_text: Label text from front QS region
            - qs_front_val_text: Value from front QS region
            - qs_back_text: Label text from back QS region
            - qs_back_val_text: Value from back QS region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file', 'qs_front_text', 'qs_front_val_text', 'qs_back_text', 'qs_back_val_text']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)

            q_coords = {k: [v[0]-1, v[1]+1, v[2]-1, v[3]+1] for k, v in qs_coords[img_size].items()}
            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    q_coords = {k: [v[0]-1, v[1]+1, v[2]-1, v[3]+1] for k, v in qs_coords2[img_size].items()}
            

            pat_metrics = [img_file]

            qs_front_coor = q_coords['qs_front']
            qs_front_val_coor = q_coords['qs_front_val']
            qs_back_coor = q_coords['qs_back']
            qs_back_val_coor = q_coords['qs_back_val']

            qs_front = img[qs_front_coor[0]-1:qs_front_coor[1]+1, qs_front_coor[2]-1:qs_front_coor[3]]
            qs_front_val = img[qs_front_val_coor[0]:qs_front_val_coor[1], qs_front_val_coor[2]:qs_front_val_coor[3]]
            qs_back = img[qs_back_coor[0]-1:qs_back_coor[1]+1, qs_back_coor[2]-1:qs_back_coor[3]]
            qs_back_val = img[qs_back_val_coor[0]:qs_back_val_coor[1], qs_back_val_coor[2]:qs_back_val_coor[3]]

            # Save temporary files
            qs_front_path = f'{temp_dir}/qs_front_{os.path.basename(img_file)}'
            qs_front_val_path = f'{temp_dir}/qs_front_val_{os.path.basename(img_file)}'
            qs_back_path = f'{temp_dir}/qs_back_{os.path.basename(img_file)}'
            qs_back_val_path = f'{temp_dir}/qs_back_val_{os.path.basename(img_file)}'

            cv2.imwrite(qs_front_path, qs_front)
            cv2.imwrite(qs_front_val_path, qs_front_val)
            cv2.imwrite(qs_back_path, qs_back)
            cv2.imwrite(qs_back_val_path, qs_back_val)

            
            qs_front_text = extract_text(qs_front_path)
            qs_front_val_text = extract_text(qs_front_val_path)
            qs_back_text = extract_text(qs_back_path)
            qs_back_val_text = extract_text(qs_back_val_path)

            # Clean up temporary file
            try:
                os.remove(qs_front_path)
                os.remove(qs_front_val_path)
                os.remove(qs_back_path)
                os.remove(qs_back_val_path)
            except:
                pass

            # print(qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text)

            pat_metrics.extend([qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_QS_google(img_files, img_size):
    """
    Extracts QS (Quality Specification) text from diagnostic images using Google Cloud Vision API.
    
    Similar to text_extraction_QS(), but uses Google Cloud Vision API for text extraction
    instead of Tesseract OCR. This may provide better accuracy in some cases, especially
    for challenging text or low-quality images.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        
    Returns:
        pandas.DataFrame: DataFrame containing QS information with columns:
            - img_file: Path to the source image
            - qs_front_text: Label text from front QS region
            - qs_front_val_text: Value from front QS region
            - qs_back_text: Label text from back QS region
            - qs_back_val_text: Value from back QS region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file', 'qs_front_text', 'qs_front_val_text', 'qs_back_text', 'qs_back_val_text']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            
            q_coords = {k: v for k, v in qs_coords[img_size].items()}
            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    q_coords = {k: v for k, v in qs_coords2[img_size].items()}

            pat_metrics = [img_file]

            qs_front_coor = q_coords['qs_front']
            qs_front_val_coor = q_coords['qs_front_val']
            qs_back_coor = q_coords['qs_back']
            qs_back_val_coor = q_coords['qs_back_val']

            qs_front = img[qs_front_coor[0]-1:qs_front_coor[1]+1, qs_front_coor[2]-1:qs_front_coor[3]]
            qs_front_val = img[qs_front_val_coor[0]:qs_front_val_coor[1], qs_front_val_coor[2]:qs_front_val_coor[3]]
            qs_back = img[qs_back_coor[0]-1:qs_back_coor[1]+1, qs_back_coor[2]-1:qs_back_coor[3]]
            qs_back_val = img[qs_back_val_coor[0]:qs_back_val_coor[1], qs_back_val_coor[2]:qs_back_val_coor[3]]

            # Save temporary files
            qs_front_path = f'{temp_dir}/qs_front_{os.path.basename(img_file)}'
            qs_front_val_path = f'{temp_dir}/qs_front_val_{os.path.basename(img_file)}'
            qs_back_path = f'{temp_dir}/qs_back_{os.path.basename(img_file)}'
            qs_back_val_path = f'{temp_dir}/qs_back_val_{os.path.basename(img_file)}'

            cv2.imwrite(qs_front_path, qs_front)
            cv2.imwrite(qs_front_val_path, qs_front_val)
            cv2.imwrite(qs_back_path, qs_back)
            cv2.imwrite(qs_back_val_path, qs_back_val)

            qs_front_data = detect_text(qs_front_path)
            qs_front_val_data = detect_text(qs_front_val_path)
            qs_back_data = detect_text(qs_back_path)
            qs_back_val_data = detect_text(qs_back_val_path)

            qs_front_data_lst = sort_by_cor(qs_front_data)
            qs_front_val_data_lst = sort_by_cor(qs_front_val_data)
            qs_back_data_lst = sort_by_cor(qs_back_data)
            qs_back_val_data_lst = sort_by_cor(qs_back_val_data)

            qs_front_text = ' '.join(qs_front_data_lst)
            qs_front_val_text = ' '.join(qs_front_val_data_lst)
            qs_back_text = ' '.join(qs_back_data_lst)
            qs_back_val_text = ' '.join(qs_back_val_data_lst)

            # Clean up temporary file
            try:
                os.remove(qs_front_path)
                os.remove(qs_front_val_path)
                os.remove(qs_back_path)
                os.remove(qs_back_val_path)
            except:
                pass

            # print(qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text)

            pat_metrics.extend([qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_QS_azure(img_files, img_size):
    """
    Extracts QS (Quality Specification) text from diagnostic images using Azure Computer Vision API.
    
    Similar to text_extraction_QS(), but uses Azure's Computer Vision API for text extraction.
    This version also includes additional image preprocessing steps like padding and may provide
    better results for certain types of images.
    
    Args:
        img_files (list): List of image file paths to process
        img_size (tuple): Tuple of (height, width) specifying image dimensions
        
    Returns:
        pandas.DataFrame: DataFrame containing QS information with columns:
            - img_file: Path to the source image
            - qs_front_text: Label text from front QS region
            - qs_front_val_text: Value from front QS region
            - qs_back_text: Label text from back QS region
            - qs_back_val_text: Value from back QS region
            
    Raises:
        ValueError: If the provided img_size is not supported
    """
    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if img_size not in qs_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    columns = ['img_file', 'qs_front_text', 'qs_front_val_text', 'qs_back_text', 'qs_back_val_text']


    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            
            q_coords = {k: v for k, v in qs_coords[img_size].items()}
            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    q_coords = {k: v for k, v in qs_coords2[img_size].items()}

            pat_metrics = [img_file]

            qs_front_coor = q_coords['qs_front']
            qs_front_val_coor = q_coords['qs_front_val']
            qs_back_coor = q_coords['qs_back']
            qs_back_val_coor = q_coords['qs_back_val']

            qs_front = img[qs_front_coor[0]-1:qs_front_coor[1]+1, qs_front_coor[2]-1:qs_front_coor[3]]
            qs_front_val = img[qs_front_val_coor[0]:qs_front_val_coor[1], qs_front_val_coor[2]:qs_front_val_coor[3]]
            qs_back = img[qs_back_coor[0]-1:qs_back_coor[1]+1, qs_back_coor[2]-1:qs_back_coor[3]]
            qs_back_val = img[qs_back_val_coor[0]:qs_back_val_coor[1], qs_back_val_coor[2]:qs_back_val_coor[3]]

            qs_concat = concat_images_vertical_with_spacing([qs_front, qs_front_val, qs_back, qs_back_val],
                                                            spacing=5)
            qs_concat = pad_image_to_min_size(qs_concat)

            


            # Save temporary files
            qs_concat_path = f'{temp_dir}/qs_concat_{os.path.basename(img_file)}'

            cv2.imwrite(qs_concat_path, qs_concat)

            qs_concat_data, _ = detect_text_azure(qs_concat_path)
            qs_concat_data_lst = sort_by_cor(qs_concat_data)

            if len(qs_concat_data_lst) < 4:
                qs_concat_data_lst = qs_concat_data_lst + [None] * (4 - len(qs_concat_data_lst))

            # Clean up temporary file
            try:
                os.remove(qs_concat_path)
            except:
                pass

            qs_front_text = qs_concat_data_lst[0]
            qs_front_val_text = qs_concat_data_lst[1]
            qs_back_text = qs_concat_data_lst[2]
            qs_back_val_text = qs_concat_data_lst[3]
            
            # print(qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text)

            pat_metrics.extend([qs_front_text, qs_front_val_text, qs_back_text, qs_back_val_text])

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df


######## TEC.7 ########################################
 
def clean_text(text):
    """
    Cleans and standardizes text by removing special characters and applying replacements.
    
    This function performs several text cleaning operations:
    1. Removes all non-alphabetic characters except spaces
    2. Normalizes whitespace
    3. Removes specified drop words
    4. Applies replacements from a predefined dictionary
    
    Args:
        text (str or other): Input text to clean. If not a string, returns unchanged.
        
    Returns:
        str or np.nan: Cleaned text string, or np.nan if the text matches a null replacement
    """
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = ' '.join([word for word in text.split(' ') if word not in drop_words])

    for old, new in replace_dict.items():
        if pd.isna(new):
            if text == old:
                return np.nan
        else:
            text = text.replace(old, new)
    
    return text

def cleaning_map_titles(ip_df, title_columns):
    """
    Cleans map titles in a DataFrame by applying text cleaning to specified columns.
    
    This function applies the clean_text function to each specified column in the
    DataFrame, standardizing the text format and handling special cases.
    
    Args:
        ip_df (pandas.DataFrame): Input DataFrame containing map titles
        title_columns (list): List of column names containing map titles to clean
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned map title columns
    """
    map_headers = title_columns
    for head_col in map_headers:
        ip_df[head_col] = ip_df[head_col].apply(clean_text)
    return ip_df

def move_qs_columns(df, from_cols, to_cols, pattern):
    """
    Moves values between columns based on pattern matching.
    
    This function identifies values in source columns that match a given pattern
    and moves them to target columns. It's particularly useful for reorganizing
    QS (Quality Specification) data where values may be in incorrect columns.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        from_cols (list): Source column names
        to_cols (list): Target column names
        pattern (str): Regex pattern to match for moving values
        
    Returns:
        pandas.DataFrame: DataFrame with values moved according to pattern matching
        
    Prints:
        - Number of pattern matches
        - Initial and final null counts in source columns
    """
    print('------------')
   
    # Create mask based on pattern matching
    mask = df[from_cols[0]].str.contains(pattern, na=False)
    
    # Print diagnostic information
    print(f'Pattern matches: {mask.value_counts()}')
    print(f'Initial nulls in {from_cols[0]}: {df[from_cols[0]].isna().sum()}')
    
    # Move values from source to target columns
    df.loc[mask, to_cols] = df.loc[mask, from_cols]
    df.loc[mask, from_cols[0]] = np.nan
    
    # Print final null count
    print(f'Final nulls in {from_cols[0]}: {df[from_cols[0]].isna().sum()}')
    
    return df

def clean_qs_columns(ip_df):
    """
    Cleans and reorganizes QS (Quality Specification) columns in a DataFrame.
    
    This function applies a series of pattern-based moves to ensure numeric and
    text values are in their correct respective columns. It handles both front
    and back QS measurements.
    
    Args:
        ip_df (pandas.DataFrame): Input DataFrame containing QS columns
        
    Returns:
        pandas.DataFrame: DataFrame with reorganized QS columns where:
            - Numeric values are in value columns
            - Text labels are in text columns
    """
    qs_cols = ['qs_front_text', 'qs_front_val_text', 'qs_back_text', 'qs_back_val_text']

    ip_df = move_qs_columns(ip_df, qs_cols[0:3], qs_cols[1:4], r'\d')
    ip_df = move_qs_columns(ip_df, qs_cols[1:3], qs_cols[2:4], r'[a-zA-Z]')
    ip_df = move_qs_columns(ip_df, qs_cols[2:3], qs_cols[3:4], r'\d')

    return ip_df

def clean_text_qs(text):
    """
    Cleans text specifically for QS (Quality Specification) fields.
    
    This function applies specialized cleaning for QS text fields, including:
    1. Dictionary-based replacements specific to QS terminology
    2. Removal of non-alphabetic characters
    3. Whitespace normalization
    
    Args:
        text (str or other): Input text to clean. If not a string, returns unchanged.
        
    Returns:
        str or np.nan: Cleaned text string, or np.nan if the text matches a null replacement
    """
    if not isinstance(text, str):
        return text
    
    # First apply dictionary replacements
    for old, new in replace_dict_qs.items():
        if pd.isna(new):
            if text == old:
                return np.nan
        else:
            if text == old:
                text = new
    
    # Then clean up remaining text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Apply dictionary replacements again
    for old, new in replace_dict_qs.items():
        if pd.isna(new):
            if text == old:
                return np.nan
        else:
            if text == old:
                text = new
    
    return text

def clean_numeric_qs(text):
    """
    Cleans numeric values from QS (Quality Specification) text fields.
    
    This function extracts and formats numeric values from text, handling:
    1. Removal of non-numeric characters (except decimal points and minus signs)
    2. Proper formatting of decimal numbers
    3. Standardization of negative numbers
    
    Args:
        text (str or other): Input text to clean. If not a string, returns unchanged.
        
    Returns:
        str or np.nan: Cleaned numeric string, or np.nan if invalid/empty
    """
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'[^0-9\-\.]', '', text)
    if text.startswith('.'):
        text = text[1:]
    
    if '-' in text:
        text = '-' + text.replace('-', '')

    if not text or text == '-':
        return np.nan

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def cleaning_qs_fields(ip_df):
    """
    Applies comprehensive cleaning to QS (Quality Specification) fields in a DataFrame.
    
    This function performs a series of cleaning operations:
    1. Reorganizes QS columns using clean_qs_columns
    2. Cleans text fields using clean_text_qs
    3. Cleans numeric fields using clean_numeric_qs
    4. Converts numeric fields to float type
    
    Args:
        ip_df (pandas.DataFrame): Input DataFrame containing QS fields
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned and properly formatted QS fields
        
    Prints:
        Value counts for front and back text fields after cleaning
    """
    ip_df = clean_qs_columns(ip_df)

    ip_df['qs_front_text'] = ip_df['qs_front_text'].apply(clean_text_qs)
    print(ip_df['qs_front_text'].value_counts())

    ip_df['qs_back_text'] = ip_df['qs_back_text'].apply(clean_text_qs)
    print(ip_df['qs_back_text'].value_counts())

    ip_df['qs_front_val_text'] = ip_df['qs_front_val_text'].apply(clean_numeric_qs).astype(float)
    ip_df['qs_back_val_text'] = ip_df['qs_back_val_text'].apply(clean_numeric_qs).astype(float)

    return ip_df


def replace_faulty_mrno(df, ref_df):

    mrn_only_letters = df[df['MRNumber'].str.match(r'^[A-Za-z-]+$', na=False)].reset_index(drop=True)
    print('cases with only letters in MRNumber',mrn_only_letters.shape)

    compare(mrn_only_letters['image_name'], ref_df['image_name'])

    found_mrns = ref_df[ref_df['image_name'].isin(mrn_only_letters['image_name'])]
    print('found images in LVP Inv dataset',found_mrns.shape)

    found_mrns = found_mrns[['image_name', 'tp_mrno']]
    found_mrns.columns = ['image_name', 'mrno']

    df = pd.merge(df, found_mrns, on='image_name', how='left')
    df['MRNumber'] = np.where(df['mrno'].isna(),
                              df['MRNumber'],
                              df['mrno'])
    df = df.drop(columns=['mrno'])

    return df

def re_clean(df):
    """
    Cleans numeric values in a standardized way.
    
    This function performs basic numeric cleaning:
    1. Replaces commas with decimal points
    2. Removes all non-numeric characters (except decimal points and minus signs)
    
    Args:
        df (str): Input string to clean
        
    Returns:
        str or np.nan: Cleaned numeric string, or np.nan if invalid/empty
    """
    a = df
    a = a.replace(',','.')
    a = re.sub("[^0-9^.-]", "", a)
    if a in [".",""]:
        return np.nan
    else:
        return a
    
def range_check(df, max_val):
    """
    Adjusts numeric values that exceed a maximum threshold.
    
    This function converts values to float and adjusts any values that exceed
    the maximum threshold by subtracting an appropriate power of 10.
    
    Args:
        df (pandas.Series): Series of numeric values to check
        max_val (float): Maximum allowed value
        
    Returns:
        pandas.Series: Series with adjusted values
    """
    df = df.astype('float')
    df = np.where(df > max_val, df - 10**(len(str(max_val))-1), df)
    return df

def clean_extractions(extract_df):
    """
    Performs comprehensive cleaning of extracted data.
    
    This function applies various cleaning operations to a DataFrame of extracted data:
    1. Converts all values to string type initially
    2. Cleans numeric columns using re_clean
    3. Applies range checks to specific measurement columns
    4. Converts cleaned numeric columns to float
    5. Standardizes date formats
    6. Handles special cases and null values
    
    Args:
        extract_df (pandas.DataFrame): DataFrame containing extracted data
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with proper data types and formats
        
    Prints:
        Shape of input DataFrame
    """
    print(extract_df.shape)
    extract_df = extract_df.astype('str')

    exempted_cols = ['name', 'MRNumber', 'dob', 'eye', 
                     'exam_date','time', 
                     'image_name', 'img_dim', 'image_path', 'header',
                     'header1', 'header2', 'header3', 'header4', 
                     'qs_front_text', 'qs_back_text']
    
    cols_to_clean = extract_df.columns.difference(exempted_cols)

    if len(cols_to_clean) != 0:
        for col in cols_to_clean:
            extract_df[col] = extract_df[col].apply(re_clean)

        extract_df[cols_to_clean] = extract_df[cols_to_clean].apply(pd.to_numeric, errors='coerce')

    extract_df['dob'] = extract_df['dob'].str.replace(".", "-")
    extract_df['exam_date'] = extract_df['exam_date'].str.replace(".", "-")

    extract_df['dob']  = pd.to_datetime(extract_df['dob'] , format="%d-%m-%Y")
    extract_df['exam_date']  = pd.to_datetime(extract_df['exam_date'] , format="%d-%m-%Y")
    extract_df['MRNumber'] = extract_df['MRNumber'].replace('Date',np.nan)
    extract_df = extract_df.dropna(subset=['MRNumber']).reset_index(drop=True)
    extract_df['MRNumber'] = extract_df['MRNumber'].str.upper().str.strip()
    extract_df['MRNumber'] = extract_df['MRNumber'].apply(lambda x: x.replace('-',''))

    extract_df = extract_df.replace('nan', np.nan)

    return extract_df

def clean_pentacam_dates(ip_df, date_col):
    """
    Standardizes date formats in Pentacam data.
    
    This function handles date formatting specifically for Pentacam data:
    1. Converts dates to string format
    2. Standardizes date separators to forward slashes
    3. Handles special cases for birth dates
    
    Args:
        ip_df (pandas.DataFrame): Input DataFrame
        date_col (str): Name of the date column to clean
        
    Returns:
        pandas.DataFrame: DataFrame with standardized date formats
    """
    ip_df[date_col] = ip_df[date_col].astype(str)
    ip_df[date_col] = ip_df[date_col].apply(lambda x:x.replace('-', '/'))
    ip_df[date_col] = ip_df[date_col].replace('nan', np.nan)

    if date_col == 'dob':
        ip_df[date_col] = ip_df[date_col].replace('00/00/2000', '01/01/2000')

    return ip_df

def pentacam_date_formatting(df):
    """
    Determines the correct date format for Pentacam dates.
    
    This function attempts to parse dates in both MM/DD/YYYY and DD/MM/YYYY formats,
    choosing the most appropriate format based on the consistency between exam date
    and date of birth formats.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'Exam Date' and 'D.o.Birth' columns
        
    Returns:
        pandas.Series: Series of properly formatted dates
    """
    default_format = '%m/%d/%Y'
    other_format = '%d/%m/%Y'
    
    # Try parsing both columns with both formats
    exam_date_default = pd.to_datetime(df['exam_date'], format=default_format, errors='coerce')
    exam_date_other = pd.to_datetime(df['exam_date'], format=other_format, errors='coerce')
    dob_default = pd.to_datetime(df['dob'], format=default_format, errors='coerce')
    
    # Create a mask for where default format works for both columns
    # default_mask = ~pd.isna(exam_date_default) & ~pd.isna(dob_default)
    default_mask = not pd.isna(exam_date_default) and not pd.isna(dob_default)
    
    # Combine results based on mask
    return pd.Series(np.where(default_mask, exam_date_default, exam_date_other))


def clean_extractions_pentacam(extract_df):
    """
    Performs comprehensive cleaning of extracted data.
    
    This function applies various cleaning operations to a DataFrame of extracted data:
    1. Converts all values to string type initially
    2. Cleans numeric columns using re_clean
    3. Applies range checks to specific measurement columns
    4. Converts cleaned numeric columns to float
    5. Standardizes date formats
    6. Handles special cases and null values
    
    Args:
        extract_df (pandas.DataFrame): DataFrame containing extracted data
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with proper data types and formats
        
    Prints:
        Shape of input DataFrame
    """
    print(extract_df.shape)
    extract_df = extract_df.astype('str')

    exempted_cols = ['name', 'MRNumber', 'dob', 'eye', 
                     'exam_date','time', 
                     'image_name', 'img_dim', 'image_path', 'header',
                     'header1', 'header2', 'header3', 'header4', 
                     'qs_front_text', 'qs_back_text']
    
    cols_to_clean = extract_df.columns.difference(exempted_cols)

    if len(cols_to_clean) != 0:
        for col in cols_to_clean:
            extract_df[col] = extract_df[col].apply(re_clean)

        extract_df[cols_to_clean] = extract_df[cols_to_clean].apply(pd.to_numeric, errors='coerce')

    # extract_df['dob']  = pd.to_datetime(extract_df['dob'] , format="%m/%d/%Y", errors='coerce')
    extract_df['MRNumber'] = extract_df['MRNumber'].replace('Date',np.nan)
    extract_df = extract_df.dropna(subset=['MRNumber']).reset_index(drop=True)
    extract_df['MRNumber'] = extract_df['MRNumber'].str.upper().str.strip()
    extract_df['MRNumber'] = extract_df['MRNumber'].apply(lambda x: x.replace('-',''))

    extract_df = extract_df.replace('nan', np.nan)

    return extract_df



########## TEC.8 #################################

def text_extraction_pentacam_google_single_shot(img_files, img_size):
   
    # Pre-compile regex patterns
    non_numeric_pattern = re.compile("[^0-9-.]")
    
    # Common column names for all image sizes
    columns = ['img_file', 'name', 'MRNumber', 'dob','eye','exam_date','time',
               'Rf_cf', 'k1_cf','Rs_cf', 'k2_cf', 'Rm_cf', 'Km_cf', 
               'Axis_cf', 'Astig_cf', 'Rper_cf', 'Rmin_cf',
               'Rf_cb', 'k1_cb','Rs_cb', 'k2_cb', 'Rm_cb', 'Km_cb', 
               'Axis_cb', 'Astig_cb', 'Rper_cb', 'Rmin_cb',
               'pupil_center', 'pc_x', 'pc_y', 'pachy_apex', 'pa_x', 'pa_y',
               'thinnest_loc', 'tl_x', 'tl_y', 'kmax_front', 'kmax_x', 'kmax_y',
               'cornea_volume', 'HWTW', 'chamber_vol', 'angle', 'ac_dept_int', 'pupil_dia']

     # Image region coordinates for different sizes
        
    region_coords2 = {

        (910, 1200): {
            'pat_info': (116, 259, 9, 327),
            'cornea_front': (273, 448, 96, 326),
            'cornea_back': (462, 637, 95, 326),
            'pachy': (665, 777, 129, 326),
            'others': (778, 878, 10, 327),
            'white_regions': {
                'cornea_front': [(51, 79, 0, 8)],
                'cornea_back': [(51, 79, 0, 8), (152, 165, 148, 158)]
            }
        }
        
        }

    region_coords = {
        (740, 1200): {
            'pat_info': (43, 157, 6, 258),
            'cornea_front': (171, 311, 80, 257),
            'cornea_back': (319, 454, 80, 257),
            'pachy': (474, 564, 101, 258),
            'others': (565, 662, 5, 257),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (119, 128, 117, 125)],
                'pachy': [(6, 85, 57, 66)]
            }
        },

        (838, 1200): {
            'pat_info': (44, 187, 8, 326),
            'cornea_front': (202, 377, 94, 326),
            'cornea_back': (391, 565, 94, 327),
            'pachy': (595, 705, 130, 325),
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
            'pachy': (638, 739, 118, 294),
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
            'pachy': (661, 770, 130, 325),
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
            'pachy': (594, 704, 129, 325),
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
            'pachy': (697, 804, 131, 325),
            'others': (809, 906, 10, 327),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (150, 160, 145, 155)]
            }
        }
    }


    if img_size not in region_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)

    # Common character lists for filtering
    char_list2 = frozenset(["",".",':','|','O',')','::','o','0','μm','(','D','mm³','mm²','*','mm','[',']','+','flt','<','flat',
                             '>>',"'",'°','.:','steep',';'])
    pat_info_drop = frozenset(['Last','Name','First','Name'])

    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            coords = region_coords[img_size]

             # Extract regions based on coordinates
            regions = {
                'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                              coords['pat_info'][2]:coords['pat_info'][3]],
                'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                  coords['cornea_front'][2]:coords['cornea_front'][3]],
                'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                 coords['cornea_back'][2]:coords['cornea_back'][3]],
                'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                           coords['pachy'][2]:coords['pachy'][3]],
                'others': img[coords['others'][0]:coords['others'][1], 
                            coords['others'][2]:coords['others'][3]]
            }

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    coords = region_coords2[img_size]
                    regions = {
                            'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                                        coords['pat_info'][2]:coords['pat_info'][3]],
                            'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                            coords['cornea_front'][2]:coords['cornea_front'][3]],
                            'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                            coords['cornea_back'][2]:coords['cornea_back'][3]],
                            'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                                    coords['pachy'][2]:coords['pachy'][3]],
                            'others': img[coords['others'][0]:coords['others'][1], 
                                        coords['others'][2]:coords['others'][3]]
                        }
                                
                    
            pat_metrics = [img_file]

            # Apply white regions
            for region, white_areas in coords['white_regions'].items():
                for y1, y2, x1, x2 in white_areas:
                    regions[region][y1:y2, x1:x2] = 255

                        
            img_concat = concat_images_vertical_with_spacing_for_metrics([region for name, region in regions.items()],
                                                                        spacing=5)
            # print(img_concat.shape)
            img_concat_path = f'{temp_dir}/img_concat_{os.path.basename(img_file)}'
            cv2.imwrite(img_concat_path, img_concat)

            img_concat_data = detect_text(img_concat_path)
          
            region_indices = get_region_indices(img_concat_data, regions)
            region_wise_data = {}
            for region_name, indices in region_indices.items():
                
                text = [img_concat_data[0][i] for i in indices]
                text_coords = [img_concat_data[1][i] for i in indices]
                
                region_wise_data[region_name] = [text, text_coords]

            data = {name: sort_by_cor(data) 
                    for name, data in region_wise_data.items()}

            try:
                os.remove(img_concat_path)
            except:
                pass

            # Process patient info
            pat_info_lst = [x.replace("|", "") for x in data['pat_info'] 
                           if x not in char_list2]
            pat_info_lst_new = [x for x in pat_info_lst if x not in pat_info_drop]
            
            try:
                id_index = pat_info_lst_new.index('ID')
                pat_name = ' '.join(pat_info_lst_new[0:id_index])

                if pat_info_lst_new[id_index+2] == '-':
                    mrn = ''.join(str(x) for x in pat_info_lst_new[(id_index+1):(id_index+4)])
                else:
                    mrn = str(pat_info_lst_new[id_index + 1])
                    
                pat_info_lst_fi = [
                    pat_name,
                    mrn,
                    pat_info_lst_new[pat_info_lst_new.index('Birth') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Eye') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Exam') + 2],
                    pat_info_lst_new[pat_info_lst_new.index('Time') + 1]
                ]                
                # Handle special eye cases
                if len([x for x in ['Left', 'Right'] if x not in pat_info_lst_fi]) == 2:
                    ind = [pat_info_lst_new.index(x) for x in ['Left', 'Right'] 
                          if x in pat_info_lst_new]
                    if ind:
                        pat_info_lst_fi[3] = pat_info_lst_new[ind[0]]
                
                pat_metrics.extend(pat_info_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            # Process cornea regions
            for region in ['cornea_front', 'cornea_back']:
                try:
                    region_lst = [x.replace("|", "") for x in data[region] 
                                if x not in char_list2]
                    
                    check1 = next((x for x in ["Rt", "Rf", "R", "Rh"] if x in region_lst), "R")
                    check2 = next((x for x in ["Rs", "Rv", "R"] if x in region_lst), "R")
                    res = [ele for ele in region_lst if "min" in ele]
                    
                    region_lst_fi = [
                        region_lst[region_lst.index(check1) + 1],
                        region_lst[region_lst.index('K1') + 1],
                        region_lst[region_lst.index(check2) + 1],
                        region_lst[region_lst.index('K2') + 1],
                        region_lst[region_lst.index('Rm') + 1],
                        region_lst[region_lst.index('Km') + 1],
                        region_lst[region_lst.index('Axis') + 1],
                        region_lst[region_lst.index('Astig') + 1],
                        region_lst[region_lst.index('Rper') + 1],
                        region_lst[region_lst.index(res[0]) + 1]
                    ]
                    pat_metrics.extend(region_lst_fi)
                except Exception as e:
                    pat_metrics.extend([''] * 10)

            # Process pachy data
            try:
                pachy_lst = [non_numeric_pattern.sub("", x) for x in data['pachy']]
                pachy_lst = [x for x in pachy_lst if x not in char_list2]
                
                if len(pachy_lst) <= 9:
                    pachy_lst = ['nan', 'nan', 'nan'] + [str(x) for x in pachy_lst]
                
                if len(pachy_lst) < 12:
                    for i, loc in enumerate([0, 3, 6, 9]):
                        if float(pachy_lst[loc]) < 10:
                            pachy_lst.insert(loc, 'nan')
                
                pat_metrics.extend([x.replace("|", "") for x in pachy_lst[:12]])
            except Exception as e:
                pat_metrics.extend([''] * 12)

            # Process others data
            try:
                others_lst = [x.replace("|", "") for x in data['others'] 
                            if x not in char_list2]
                
                check = "Int" if "Int" in others_lst else "Ext"
                check_chamber = "Chamber" if "Chamber" in others_lst else "Chamb"
                check_pupil = "Dia" if "Dia" in others_lst else "Pupil"
                others_lst_fi = [
                    others_lst[others_lst.index('Volume') + 1],
                    others_lst[others_lst.index('HWTW') + 1],
                    others_lst[others_lst.index(check_chamber) + 2],
                    others_lst[others_lst.index('Angle') + 1],
                    others_lst[others_lst.index(check) + 1],
                    others_lst[others_lst.index(check_pupil) + 1]
                ]
                pat_metrics.extend(others_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_pentacam_azure_single_shot(img_files, img_size):
   
    # Pre-compile regex patterns
    non_numeric_pattern = re.compile("[^0-9-.]")
    
    # Common column names for all image sizes
    columns = ['img_file', 'name', 'MRNumber', 'dob','eye','exam_date','time',
               'Rf_cf', 'k1_cf','Rs_cf', 'k2_cf', 'Rm_cf', 'Km_cf', 
               'Axis_cf', 'Astig_cf', 'Rper_cf', 'Rmin_cf',
               'Rf_cb', 'k1_cb','Rs_cb', 'k2_cb', 'Rm_cb', 'Km_cb', 
               'Axis_cb', 'Astig_cb', 'Rper_cb', 'Rmin_cb',
               'pupil_center', 'pc_x', 'pc_y', 'pachy_apex', 'pa_x', 'pa_y',
               'thinnest_loc', 'tl_x', 'tl_y', 'kmax_front', 'kmax_x', 'kmax_y',
               'cornea_volume', 'HWTW', 'chamber_vol', 'angle', 'ac_dept_int', 'pupil_dia']

     # Image region coordinates for different sizes
        
    region_coords2 = {

        (910, 1200): {
            'pat_info': (116, 259, 9, 327),
            'cornea_front': (273, 448, 96, 326),
            'cornea_back': (462, 637, 95, 326),
            'pachy': (665, 777, 129, 326),
            'others': (778, 878, 10, 327),
            'white_regions': {
                'cornea_front': [(51, 79, 0, 8)],
                'cornea_back': [(51, 79, 0, 8), (152, 165, 148, 158)]
            }
        }
        
        }

    region_coords = {
        (740, 1200): {
            'pat_info': (43, 157, 6, 258),
            'cornea_front': (171, 311, 80, 257),
            'cornea_back': (319, 454, 80, 257),
            'pachy': (474, 564, 101, 258),
            'others': (565, 662, 5, 257),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (119, 128, 117, 125)],
                'pachy': [(6, 85, 57, 66)]
            }
        },

        (838, 1200): {
            'pat_info': (44, 187, 8, 326),
            'cornea_front': (202, 377, 94, 326),
            'cornea_back': (391, 565, 94, 327),
            'pachy': (595, 705, 130, 325),
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
            'pachy': (638, 739, 118, 294),
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
            'pachy': (661, 770, 130, 325),
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
            'pachy': (594, 704, 129, 325),
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
            'pachy': (697, 804, 131, 325),
            'others': (809, 906, 10, 327),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (150, 160, 145, 155)]
            }
        }
    }


    if img_size not in region_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)

    # Common character lists for filtering
    char_list2 = frozenset(["",".",':','|','O',')','::','o','0','μm','(','D','mm³','mm²','*','mm','[',']','+','flt','<','flat',
                             '>>',"'",'°','.:','steep',';'])
    pat_info_drop = frozenset(['Last','Name','First','Name'])

    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            coords = region_coords[img_size]

             # Extract regions based on coordinates
            regions = {
                'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                              coords['pat_info'][2]:coords['pat_info'][3]],
                'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                  coords['cornea_front'][2]:coords['cornea_front'][3]],
                'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                 coords['cornea_back'][2]:coords['cornea_back'][3]],
                'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                           coords['pachy'][2]:coords['pachy'][3]],
                'others': img[coords['others'][0]:coords['others'][1], 
                            coords['others'][2]:coords['others'][3]]
            }

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    coords = region_coords2[img_size]
                    regions = {
                            'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                                        coords['pat_info'][2]:coords['pat_info'][3]],
                            'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                            coords['cornea_front'][2]:coords['cornea_front'][3]],
                            'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                            coords['cornea_back'][2]:coords['cornea_back'][3]],
                            'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                                    coords['pachy'][2]:coords['pachy'][3]],
                            'others': img[coords['others'][0]:coords['others'][1], 
                                        coords['others'][2]:coords['others'][3]]
                        }
                                
                    
            pat_metrics = [img_file]

            # Apply white regions
            for region, white_areas in coords['white_regions'].items():
                for y1, y2, x1, x2 in white_areas:
                    regions[region][y1:y2, x1:x2] = 255

                        
            img_concat = concat_images_vertical_with_spacing_for_metrics([region for name, region in regions.items()],
                                                                        spacing=5)
            # print(img_concat.shape)
            img_concat_path = f'{temp_dir}/img_concat_{os.path.basename(img_file)}'
            cv2.imwrite(img_concat_path, img_concat)

            _, img_concat_data = detect_text_azure(img_concat_path)
          
            region_indices = get_region_indices(img_concat_data, regions)
            region_wise_data = {}
            for region_name, indices in region_indices.items():
                
                text = [img_concat_data[0][i] for i in indices]
                text_coords = [img_concat_data[1][i] for i in indices]
                
                region_wise_data[region_name] = [text, text_coords]

            data = {name: sort_by_cor(data) 
                    for name, data in region_wise_data.items()}

            try:
                os.remove(img_concat_path)
            except:
                pass

            # Process patient info
            pat_info_lst = [x.replace("|", "") for x in data['pat_info'] 
                            if x not in char_list2]
            pat_info_lst_new = [x.replace(':', '') for x in pat_info_lst]
            pat_info_lst_new = [x for x in pat_info_lst_new if x not in pat_info_drop]
                        
            try:
                id_index = pat_info_lst_new.index('ID')
                pat_name = ' '.join(pat_info_lst_new[0:id_index])
                
                if pat_info_lst_new[id_index+2] == '-':
                    mrn = ''.join(str(x) for x in pat_info_lst_new[(id_index+1):(id_index+4)])
                else:
                    mrn = str(pat_info_lst_new[id_index + 1])

                check_eye = "Eye" if "Eye" in pat_info_lst_new else "Eve"
                    
                pat_info_lst_fi = [
                    pat_name,
                    mrn,
                    pat_info_lst_new[pat_info_lst_new.index('Birth') + 1],
                    pat_info_lst_new[pat_info_lst_new.index(check_eye) + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Exam') + 2],
                    pat_info_lst_new[pat_info_lst_new.index('Time') + 1]
                ]                
                # Handle special eye cases
                if len([x for x in ['Left', 'Right'] if x not in pat_info_lst_fi]) == 2:
                    ind = [pat_info_lst_new.index(x) for x in ['Left', 'Right'] 
                            if x in pat_info_lst_new]
                    if ind:
                        pat_info_lst_fi[3] = pat_info_lst_new[ind[0]]
                
                pat_metrics.extend(pat_info_lst_fi)
                
            except Exception as e:
                pat_metrics.extend([''] * 6)

            # Process cornea regions
            for region in ['cornea_front', 'cornea_back']:
                try:
                    region_lst = [x.replace("|", "") for x in data[region] 
                                if x not in char_list2]
                    region_lst = [x.replace(':', '') for x in region_lst]

                    check1 = next((x for x in ["Rt", "Rf", "R", "Rh", "RE", "RF"] if x in region_lst), "R")
                    check2 = next((x for x in ["Rs", "Rv", "R"] if x in region_lst), "R")
                    check_astig = 'Astig' if "Astig" in region_lst else "Astiq"
                    check_k1 = 'K1' if "K1" in region_lst else "<1"
                    res = [ele for ele in region_lst if "min" in ele]

                    region_lst_fi = [
                        region_lst[region_lst.index(check1) + 1],
                        region_lst[region_lst.index(check_k1) + 1],
                        region_lst[region_lst.index(check2) + 1],
                        region_lst[region_lst.index('K2') + 1],
                        region_lst[region_lst.index('Rm') + 1],
                        region_lst[region_lst.index('Km') + 1],
                        region_lst[region_lst.index('Axis') + 1],
                        region_lst[region_lst.index(check_astig) + 1],
                        region_lst[region_lst.index('Rper') + 1],
                        region_lst[region_lst.index(res[0]) + 1]
                    ]

                    pat_metrics.extend(region_lst_fi)
                except Exception as e:
                    pat_metrics.extend([''] * 10)

            # Process pachy data
            try:
                pachy_lst = [non_numeric_pattern.sub("", x) for x in data['pachy']]
                pachy_lst = [x for x in pachy_lst if x not in char_list2]
                
                if len(pachy_lst) <= 9:
                    pachy_lst = ['nan', 'nan', 'nan'] + [str(x) for x in pachy_lst]
                
                if len(pachy_lst) < 12:
                    for i, loc in enumerate([0, 3, 6, 9]):
                        if float(pachy_lst[loc]) < 10:
                            pachy_lst.insert(loc, 'nan')
                
                pat_metrics.extend([x.replace("|", "") for x in pachy_lst[:12]])
            except Exception as e:
                pat_metrics.extend([''] * 12)

            # Process others data
            try:
                others_lst = [x.replace("|", "") for x in data['others'] 
                            if x not in char_list2]
                others_lst = [x.replace(':', '') for x in others_lst]

                check = next((x for x in ["(Int.]", "Int", "Ext"] if x in others_lst), "Int")
                check_chamber = next((x for x in ["Chamber", "Chamb", "Chamb."] if x in others_lst), "Chamber")
                check_pupil = "Dia" if "Dia" in others_lst else "Pupil"
                others_lst_fi = [
                    others_lst[others_lst.index('Volume') + 1],
                    others_lst[others_lst.index('HWTW') + 1],
                    others_lst[others_lst.index(check_chamber) + 2],
                    others_lst[others_lst.index('Angle') + 1],
                    others_lst[others_lst.index(check) + 1],
                    others_lst[others_lst.index(check_pupil) + 1]
                ]
                pat_metrics.extend(others_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

def text_extraction_pentacam_google_multi_shot(img_files, img_size):
   
    # Pre-compile regex patterns
    non_numeric_pattern = re.compile("[^0-9-.]")
    
    # Common column names for all image sizes
    columns = ['img_file', 'name', 'MRNumber', 'dob','eye','exam_date','time',
               'Rf_cf', 'k1_cf','Rs_cf', 'k2_cf', 'Rm_cf', 'Km_cf', 
               'Axis_cf', 'Astig_cf', 'Rper_cf', 'Rmin_cf',
               'Rf_cb', 'k1_cb','Rs_cb', 'k2_cb', 'Rm_cb', 'Km_cb', 
               'Axis_cb', 'Astig_cb', 'Rper_cb', 'Rmin_cb',
               'pupil_center', 'pc_x', 'pc_y', 'pachy_apex', 'pa_x', 'pa_y',
               'thinnest_loc', 'tl_x', 'tl_y', 'kmax_front', 'kmax_x', 'kmax_y',
               'cornea_volume', 'HWTW', 'chamber_vol', 'angle', 'ac_dept_int', 'pupil_dia']

     # Image region coordinates for different sizes
        
    region_coords2 = {

        (910, 1200): {
            'pat_info': (116, 259, 9, 327),
            'cornea_front': (273, 448, 96, 326),
            'cornea_back': (462, 637, 95, 326),
            'pachy': (665, 777, 129, 326),
            'others': (778, 878, 10, 327),
            'white_regions': {
                'cornea_front': [(51, 79, 0, 8)],
                'cornea_back': [(51, 79, 0, 8), (152, 165, 148, 158)]
            }
        }
        
        }

    region_coords = {
        (740, 1200): {
            'pat_info': (43, 157, 6, 258),
            'cornea_front': (171, 311, 80, 257),
            'cornea_back': (319, 454, 80, 257),
            'pachy': (474, 564, 101, 258),
            'others': (565, 662, 5, 257),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (119, 128, 117, 125)],
                'pachy': [(6, 85, 57, 66)]
            }
        },

        (838, 1200): {
            'pat_info': (44, 187, 8, 326),
            'cornea_front': (202, 377, 94, 326),
            'cornea_back': (391, 565, 94, 327),
            'pachy': (595, 705, 130, 325),
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
            'pachy': (638, 739, 118, 294),
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
            'pachy': (661, 770, 130, 325),
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
            'pachy': (594, 704, 129, 325),
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
            'pachy': (697, 804, 131, 325),
            'others': (809, 906, 10, 327),
            'white_regions': {
                'cornea_front': [(47, 81, 0, 5)],
                'cornea_back': [(47, 81, 0, 5), (150, 160, 145, 155)]
            }
        }
    }


    if img_size not in region_coords:
        raise ValueError(f"Unsupported image size: {img_size}")

    temp_dir = f'TEC/text_ex_temp/temp_{img_size[0]}_{img_size[1]}'
    os.makedirs(temp_dir, exist_ok=True)

    # Common character lists for filtering
    char_list2 = frozenset(["",".",':','|','O',')','::','o','0','μm','(','D','mm³','mm²','*','mm','[',']','+','flt','<','flat', 
                            '>>',"'",'°','.:','steep',';'])
    pat_info_drop = frozenset(['Last','Name','First','Name'])

    def process_single_image(img_file):
        try:
            img = cv2.imread(img_file, 1)
            coords = region_coords[img_size]

             # Extract regions based on coordinates
            regions = {
                'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                              coords['pat_info'][2]:coords['pat_info'][3]],
                'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                  coords['cornea_front'][2]:coords['cornea_front'][3]],
                'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                 coords['cornea_back'][2]:coords['cornea_back'][3]],
                'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                           coords['pachy'][2]:coords['pachy'][3]],
                'others': img[coords['others'][0]:coords['others'][1], 
                            coords['others'][2]:coords['others'][3]]
            }

            if img_size == (910, 1200):
                header_coords = image_header_coord_dict['_'.join(str(x) for x in list(img_size))]
                header = img[header_coords[0]:header_coords[1], header_coords[2]:header_coords[3]]
                header_df = op_to_df(header)
                if ' '.join(header_df["Value"]) == 'DPacad FVF inctiti':
                    coords = region_coords2[img_size]
                    regions = {
                            'pat_info': img[coords['pat_info'][0]:coords['pat_info'][1], 
                                        coords['pat_info'][2]:coords['pat_info'][3]],
                            'cornea_front': img[coords['cornea_front'][0]:coords['cornea_front'][1], 
                                            coords['cornea_front'][2]:coords['cornea_front'][3]],
                            'cornea_back': img[coords['cornea_back'][0]:coords['cornea_back'][1], 
                                            coords['cornea_back'][2]:coords['cornea_back'][3]],
                            'pachy': img[coords['pachy'][0]:coords['pachy'][1], 
                                    coords['pachy'][2]:coords['pachy'][3]],
                            'others': img[coords['others'][0]:coords['others'][1], 
                                        coords['others'][2]:coords['others'][3]]
                        }
                                
                    
            pat_metrics = [img_file]

           

            # Apply white regions
            for region, white_areas in coords['white_regions'].items():
                for y1, y2, x1, x2 in white_areas:
                    regions[region][y1:y2, x1:x2] = 255

            # Save and process regions
            paths = {name: f'{temp_dir}/{name}_{os.path.basename(img_file)}' 
                    for name in regions}
            
            for name, region in regions.items():
                cv2.imwrite(paths[name], region)

            data = {name: sort_by_cor(detect_text(path)) 
                   for name, path in paths.items()}

            # Clean up temporary files
            for path in paths.values():
                try:
                    os.remove(path)
                except:
                    pass

            # Process patient info
            pat_info_lst = [x.replace("|", "") for x in data['pat_info'] 
                           if x not in char_list2]
            pat_info_lst_new = [x for x in pat_info_lst if x not in pat_info_drop]
            
            try:
                id_index = pat_info_lst_new.index('ID')
                pat_name = ' '.join(pat_info_lst_new[0:id_index])

                if pat_info_lst_new[id_index+2] == '-':
                    mrn = ''.join(str(x) for x in pat_info_lst_new[(id_index+1):(id_index+4)])
                else:
                    mrn = str(pat_info_lst_new[id_index + 1])
                    
                pat_info_lst_fi = [
                    pat_name,
                    mrn,
                    pat_info_lst_new[pat_info_lst_new.index('Birth') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Eye') + 1],
                    pat_info_lst_new[pat_info_lst_new.index('Exam') + 2],
                    pat_info_lst_new[pat_info_lst_new.index('Time') + 1]
                ]                
                # Handle special eye cases
                if len([x for x in ['Left', 'Right'] if x not in pat_info_lst_fi]) == 2:
                    ind = [pat_info_lst_new.index(x) for x in ['Left', 'Right'] 
                          if x in pat_info_lst_new]
                    if ind:
                        pat_info_lst_fi[3] = pat_info_lst_new[ind[0]]
                
                pat_metrics.extend(pat_info_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            # Process cornea regions
            for region in ['cornea_front', 'cornea_back']:
                try:
                    region_lst = [x.replace("|", "") for x in data[region] 
                                if x not in char_list2]
                    
                    check1 = next((x for x in ["Rt", "Rf", "R", "Rh"] if x in region_lst), "R")
                    check2 = next((x for x in ["Rs", "Rv", "R"] if x in region_lst), "R")
                    res = [ele for ele in region_lst if "min" in ele]
                    
                    region_lst_fi = [
                        region_lst[region_lst.index(check1) + 1],
                        region_lst[region_lst.index('K1') + 1],
                        region_lst[region_lst.index(check2) + 1],
                        region_lst[region_lst.index('K2') + 1],
                        region_lst[region_lst.index('Rm') + 1],
                        region_lst[region_lst.index('Km') + 1],
                        region_lst[region_lst.index('Axis') + 1],
                        region_lst[region_lst.index('Astig') + 1],
                        region_lst[region_lst.index('Rper') + 1],
                        region_lst[region_lst.index(res[0]) + 1]
                    ]
                    pat_metrics.extend(region_lst_fi)
                except Exception as e:
                    pat_metrics.extend([''] * 10)

            # Process pachy data
            try:
                pachy_lst = [non_numeric_pattern.sub("", x) for x in data['pachy']]
                pachy_lst = [x for x in pachy_lst if x not in char_list2]
                
                if len(pachy_lst) <= 9:
                    pachy_lst = ['nan', 'nan', 'nan'] + [str(x) for x in pachy_lst]
                
                if len(pachy_lst) < 12:
                    for i, loc in enumerate([0, 3, 6, 9]):
                        if float(pachy_lst[loc]) < 10:
                            pachy_lst.insert(loc, 'nan')
                
                pat_metrics.extend([x.replace("|", "") for x in pachy_lst[:12]])
            except Exception as e:
                pat_metrics.extend([''] * 12)

            # Process others data
            try:
                others_lst = [x.replace("|", "") for x in data['others'] 
                            if x not in char_list2]
                
                check = "Int" if "Int" in others_lst else "Ext"
                check_chamber = "Chamber" if "Chamber" in others_lst else "Chamb"
                check_pupil = "Dia" if "Dia" in others_lst else "Pupil"
                others_lst_fi = [
                    others_lst[others_lst.index('Volume') + 1],
                    others_lst[others_lst.index('HWTW') + 1],
                    others_lst[others_lst.index(check_chamber) + 2],
                    others_lst[others_lst.index('Angle') + 1],
                    others_lst[others_lst.index(check) + 1],
                    others_lst[others_lst.index(check_pupil) + 1]
                ]
                pat_metrics.extend(others_lst_fi)
            except Exception as e:
                pat_metrics.extend([''] * 6)

            return pat_metrics

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None

    # Process images in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_files),
            total=len(img_files),
            desc="Processing images"
        ))

    # Create DataFrame from valid results
    valid_results = [r for r in results if r is not None]
    pat_metrics_df = pd.DataFrame(valid_results, columns=columns)

    return pat_metrics_df

########################################################

def clean_pentacam_elevation_extractions(ele_metric_data):
    """
    Cleans elevation metrics extracted from Pentacam data.
    
    This function performs specialized cleaning for elevation metrics:
    1. Converts all values to string initially
    2. Cleans numeric columns using re_clean
    3. Handles date formatting in multiple formats
    4. Processes elevation-specific measurements
    
    Args:
        ele_metric_data (pandas.DataFrame): DataFrame containing elevation metrics
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with properly formatted elevation metrics
        
    Prints:
        Column names being cleaned and duplicate counts
    """
    ele_metric_data = ele_metric_data.astype('str')
    exempted_cols = ['img_file', 'name', 'MRNumber', 'dob', 'eye', 'exam_date','time','ele_front_mets']

    cols_to_clean = ele_metric_data.columns.difference(exempted_cols)
    print(cols_to_clean)

    for i in range(8,ele_metric_data.shape[1]):
        print(ele_metric_data.columns[i])
        ele_metric_data[ele_metric_data.columns[i]] = ele_metric_data[ele_metric_data.columns[i]].apply(re_clean)

    ele_metric_data['dob'] = ele_metric_data['dob'].str.replace(".", "-", regex=True)
    ele_metric_data['exam_date'] = ele_metric_data['exam_date'].str.replace(".", "-", regex=True)

    ele_metric_data['dob2'] = ele_metric_data['dob']
    ele_metric_data['exam_date2'] = ele_metric_data['exam_date']

    ele_metric_data['dob2']  = pd.to_datetime(ele_metric_data['dob2'] , format="%m/%d/%Y", errors= 'coerce')
    ele_metric_data['exam_date2']  = pd.to_datetime(ele_metric_data['exam_date2'] , format="%m/%d/%Y", errors= 'coerce')

    ele_metric_data['dob']  = pd.to_datetime(ele_metric_data['dob'] , format="%d/%m/%Y", errors= 'coerce')
    ele_metric_data['exam_date']  = pd.to_datetime(ele_metric_data['exam_date'] , format="%d/%m/%Y", errors= 'coerce')

    ele_metric_data['dob'] = np.where(pd.Series(ele_metric_data['dob']).isnull(), 
                                      ele_metric_data['dob2'], 
                                      ele_metric_data['dob'])

    ele_metric_data['exam_date'] = np.where(pd.Series(ele_metric_data['exam_date']).isnull(), 
                                      ele_metric_data['exam_date2'], 
                                      ele_metric_data['exam_date'])

    ele_metric_data['MRNumber'] = ele_metric_data['MRNumber'].replace('Date',np.nan)

    ele_metric_data['ele_front_max'] = ele_metric_data['ele_front_max'].str.replace("-", "", regex=True)
    ele_metric_data['ele_front_max'] = ele_metric_data['ele_front_max'].astype('float')
    print(ele_metric_data.duplicated(['img_file']).value_counts())

    return ele_metric_data

def clean_oculyzer_elevation_extractions(ele_metric_data):
    """
    Cleans elevation metrics extracted from Oculyzer data.
    
    This function performs specialized cleaning for Oculyzer elevation metrics:
    1. Converts all values to string initially
    2. Cleans numeric columns using re_clean
    3. Converts elevation measurements to float type
    
    Args:
        ele_metric_data (pandas.DataFrame): DataFrame containing Oculyzer elevation metrics
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with properly formatted elevation metrics
        
    Prints:
        Column names being cleaned and duplicate counts
    """
    ele_metric_data = ele_metric_data.astype('str')
    exempted_cols = ['img_file', 'ele_front_mets']

    cols_to_clean = ele_metric_data.columns.difference(exempted_cols)

    for i in range(2,ele_metric_data.shape[1]):
        print(ele_metric_data.columns[i])
        ele_metric_data[ele_metric_data.columns[i]] = ele_metric_data[ele_metric_data.columns[i]].apply(re_clean)

    ele_metric_data['ele_front_max'] = ele_metric_data['ele_front_max'].astype('float')
    print(ele_metric_data.duplicated(['img_file']).value_counts())
    
    return ele_metric_data

def combine_text_by_region(text_data, image_heights, spacing=20):
    """
    Combines extracted text from multiple image regions based on vertical positions.
    
    This function takes text data extracted from multiple vertically stacked image regions
    and combines them while preserving their spatial relationships. It's particularly useful
    for handling text extracted from composite images or multi-region scans.
    
    Args:
        text_data (tuple): Tuple containing two elements:
            - line_text (list): List of extracted text strings
            - line_coords (list): List of coordinate strings in "(x,y)" format
        image_heights (list): List of heights for each image region
        spacing (int, optional): Vertical spacing between regions in pixels. Defaults to 20.
        
    Returns:
        list: List of combined text strings, one for each image region, with text elements
              grouped based on their vertical positions in the original image.
    """
    line_text, line_coords = text_data
    
    # Extract y-coordinates from the first point of each coordinate string
    y_coords = []
    for coord_str in line_coords:
        # Extract y coordinate from string like "(x,y)"
        y = int(coord_str[0].split(',')[1].strip(')'))
        y_coords.append(y)
    
    # Sort text and coordinates by y-coordinate
    sorted_data = sorted(zip(y_coords, line_text), key=lambda x: x[0])
    y_coords, line_text = zip(*sorted_data)
    
    # Calculate exact boundaries for each region
    boundaries = []
    current_y = 0
    for height in image_heights:
        boundaries.append((current_y, current_y + height))
        current_y += height + spacing
    
    # Group text by region using exact boundaries
    combined_text = [''] * len(image_heights)
    for y, text in zip(y_coords, line_text):
        for i, (start, end) in enumerate(boundaries):
            if start <= y <= end:
                if combined_text[i]:
                    combined_text[i] += ' ' + text
                else:
                    combined_text[i] = text
                break
    
    return combined_text

def select_region(image):
    """
    Function to select a region on an image using mouse clicks.
    Returns the coordinates of the selected region [x1, y1, x2, y2]
    """
    # Global variables
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    coordinates = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, fx, fy, drawing, coordinates
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            coordinates = []  # Reset coordinates when starting new selection
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Create a copy of the image to draw on
                img_copy = img.copy()
                # Draw rectangle from start point to current point
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            # Draw final rectangle
            cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow('Image', img)
            
            # Store coordinates
            # coordinates = [ix, iy, fx, fy]
            coordinates = [iy,fy,ix,fx]
            print(f"Selected region coordinates: {coordinates}")
    
    # Create a copy of the input image
    img = image.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    
    # Display the image
    cv2.imshow('Image', img)
    
    print("Click and drag to select a region. Press 'q' to quit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return coordinates








