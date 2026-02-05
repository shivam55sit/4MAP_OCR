##### constants #####

import numpy as np

image_header_coord_dict = {
        '758_1200': [2, 40, 2, 948],
        '820_1200': [2, 40, 2, 948],
        '840_1200': [2, 40, 2, 948],
        '894_1600': [2, 40, 2, 948],
        '740_1200': [2, 40, 2, 948],
        '838_1200': [2, 40, 2, 948],
        '858_1200': [100, 140, 2, 948],
        '904_1200': [70, 110, 2, 948],
        '910_1200': [2, 40, 2, 948],
        '940_1200': [100, 140, 2, 948]
    }

used_img_dims = ['740_1200', '758_1200', '820_1200', '838_1200', '840_1200',
                 '858_1200', '894_1600', '904_1200', '910_1200', '940_1200']

model_headers = ['WAVELIGHT ALLEGRO OCULYZER 4 Maps Refractive', 'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable', 
                        'OCULUS PENTACAM 4 Maps Refractive', 'OCULUS PENTACAM 4 Maps Selectable']



qs_coords = {(740, 1200): {'qs_front': [265, 281, 43, 77], 'qs_front_val': [289, 305, 43, 77], 
                                'qs_back': [408, 425, 42, 75], 'qs_back_val': [434, 451, 42, 78]},

            (838, 1200): {'qs_front': [321,339,54,90], 'qs_front_val': [351,369,54,90], 
                                    'qs_back': [506,523,49,82], 'qs_back_val': [537,553,54,90]},

            (858, 1200): {'qs_front': [399,415,49,82], 'qs_front_val': [426,441,49,82], 
                                    'qs_back': [561,578,49,82], 'qs_back_val': [587,603,49,82]},

            (904, 1200): {'qs_front': [386,403,54,90], 'qs_front_val': [419,433,54,90], 
                                    'qs_back': [571,590,49,82], 'qs_back_val': [603,620,54,90]},

            (910, 1200): {'qs_front': [319, 340, 53, 94], 'qs_front_val': [351, 368, 53, 94], 
                                    'qs_back': [505, 523, 52, 93], 'qs_back_val': [538, 556, 53, 94]},

            (940, 1200): {'qs_front': [422,439,54,90], 'qs_front_val': [455,469,54,90], 
                                    'qs_back': [608,626,54,90], 'qs_back_val': [639,657,54,90]},
                                
                                
            (758, 1200): {'qs_front': [295,314, 49,84], 'qs_front_val': [323,342,49,84], 
                                'qs_back': [462,483,49,84], 'qs_back_val': [491,508,49,84]},
                                
            (820, 1200): {'qs_front': [317,334, 55,91], 'qs_front_val': [348,362,55,91], 
                                    'qs_back': [496,514,55,91], 'qs_back_val': [528,543,55,91]},
            
            (840, 1200): {'qs_front': [324,340, 55,94], 'qs_front_val': [353,369,55,94], 
                                    'qs_back': [508,524,55,94], 'qs_back_val': [540,557,55,94]},
            
            (894, 1600): {'qs_front': [339,355,59,98], 'qs_front_val': [372,388,59,98], 
                                    'qs_back': [535,552,59,98], 'qs_back_val': [569,587,59,98]}
        }

qs_coords2 = {(910, 1200): {'qs_front': [391, 413, 53, 97], 'qs_front_val': [423, 443, 54, 93], 
                            'qs_back': [577, 598, 53, 91], 'qs_back_val': [607, 630, 54, 91]}
                            }

selectable_map_header_coords = {(740, 1200): {'header1': [52,72,319,700], 'header2': [52,72,741,1123], 
                                            'header3': [363,385,319,700], 'header4': [363,385,741, 1123]},

                                (838, 1200): {'header1': [55,73,401,732], 'header2': [55,73,777,1109], 
                                            'header3': [454,473,401,732], 'header4': [454,473,777,1109]},

                                (858, 1200): {'header1': [153,173,362,717], 'header2': [153,173,760,1115], 
                                            'header3': [512,532,362,717], 'header4': [512,532,760,1115]},

                                (904, 1200): {'header1': [119,140,402,732], 'header2': [119,140,777,1109], 
                                            'header3': [519,537,402,732], 'header4': [519,537,777,1109]},
                                
                                (910, 1200): {'header1': [54,74,402,732], 'header2': [54,74,777,1107], 
                                            'header3': [452,472,402,732], 'header4': [452,472,777,1107]},

                                (940, 1200): {'header1': [154,174,401, 734], 'header2': [154,174,776,1107], 
                                            'header3': [554,573,401, 734], 'header4': [554,573,776,1107]},
                                            
                                (758, 1200): {'header1': [52,72,365,713], 'header2': [52,72,760,1114], 
                                            'header3': [414,434,365,713], 'header4': [414,434,760,1114]},
                
                                (820, 1200): {'header1': [56,76,396,729], 'header2': [56,76,776,1109], 
                                                'header3': [445,465,396,729], 'header4': [445,465,776,1109]},
                                
                                (840, 1200): {'header1': [56,76,406,732], 'header2': [56,76,782,1108], 
                                                'header3': [456,476,406,732], 'header4': [456,476,782,1108]},
                                
                                (894, 1600): {'header1': [56,76,433,932], 'header2': [56,76,993,1495], 
                                                'header3': [480,500,433,932], 'header4': [480,500,993,1495]}
                                }


refractive_map_header_coords = {(740, 1200): {'header1': [83,100,404,671], 'header2': [78,105,825,1075], 
                                            'header3': [378,396,404,671], 'header4': [375,401,825,1075]},

                                (838, 1200): {'header1': [93, 116, 424, 734], 'header2': [90, 121, 798, 1115], 
                                            'header3': [469, 491, 426, 736], 'header4': [467, 498, 803, 1113]},

                                (858, 1200): {'header1': [187, 211, 414, 697], 'header2': [187, 215, 813, 1114], 
                                            'header3': [525, 548, 418, 710], 'header4': [525, 554, 813, 1117]},

                                (904, 1200): {'header1':[157, 182, 421, 727], 'header2': [156, 186, 798, 1110], 
                                            'header3': [535, 559, 425, 737], 'header4': [532, 563, 804, 1110]},
                                
                                (910, 1200): {'header1': [92, 117, 425, 727], 'header2': [89, 121, 802, 1102], 
                                            'header3': [469, 492, 427, 738], 'header4': [468, 499, 800, 1101]},

                                (940, 1200): {'header1': [195, 219, 423, 724], 'header2': [192, 223, 799, 1103], 
                                            'header3': [569, 595, 427, 733], 'header4': [568, 600, 801, 1101]},
                                            
                                (758, 1200): {'header1': [87, 108, 420, 727], 'header2': [85, 114, 812, 1111], 
                                            'header3': [426, 449, 419, 718], 'header4': [425, 454, 813, 1109]},
                
                                (820, 1200): {'header1': [92, 114, 423, 732], 'header2': [88, 119, 805, 1116], 
                                                'header3': [458, 483, 426, 722], 'header4': [458, 489, 805, 1108]},
                                
                                (840, 1200): {'header1': [93, 117, 425, 727], 'header2': [91, 122, 802, 1116], 
                                                'header3': [469, 498, 431, 737], 'header4': [469, 502, 805, 1105]},
                                
                                (894, 1600): {'header1': [96, 124, 543, 893], 'header2': [94, 128, 1099, 1443], 
                                                'header3': [496, 524, 543, 890], 'header4': [497, 529, 1099, 1433]}
                                }

refractive_map_header_coords2 = {(910, 1200): {'header1': [164, 189, 422, 736], 'header2': [162, 195, 801, 1105], 
                                            'header3': [540, 568, 429, 736], 'header4': [537, 570, 802, 1105]}
                                }

IMG_SIZE_TO_DIMENSIONS = {
    '740_1200': (740, 1200),
    '758_1200': (758, 1200),
    '820_1200': (820, 1200),
    '840_1200': (840, 1200),
    '894_1600': (894, 1600),
    '838_1200': (838, 1200),
    '858_1200': (858, 1200),
    '904_1200': (904, 1200),
    '910_1200': (910, 1200),
    '940_1200': (940, 1200)
}


replace_dict = {'Axial Saaittal Curvature Front':'Axial Sagittal Curvature Front', 
                'Axial Saqittal Curvature Front':'Axial Sagittal Curvature Front', 
                'vial Sagittal Curvature Font':'Axial Sagittal Curvature Front',
                'Avial Sagittal Curvature Front':'Axial Sagittal Curvature Front', 
                'Aial Sacittal Curvature Front':'Axial Sagittal Curvature Front',
                'Charge Bill Jurvature Front':'Axial Sagittal Curvature Front',
                'Axial Sagittal Cun':'Axial Sagittal Curvature Front',
                'axial Sagittal Curvature Front':'Axial Sagittal Curvature Front',
                'Aial Sagittal Curvature Front':'Axial Sagittal Curvature Front',
                'SSE':'Axial Sagittal Curvature Front',
                
                'Axial Saaittal Curvature Back':'Axial Sagittal Curvature Back',
                
                'Come Thickness':'Corneal Thickness', 
                'Comneal Thickness':'Corneal Thickness', 
                'Comeal Thickness':'Corneal Thickness', 
                'Comesl Thickness':'Corneal Thickness',

                'Total Comeal Refractive Power':'Total Corneal Refractive Power',
                
                'TRENT':'', 'Ail tt':''}

drop_words = ['BF', 'BFS', 'BFSd', 'Di', 'Dia', 'Diae', 'Die', 'Dig', 'Dja', 'Flo',
               'Float', 'FloatDia', 'Floe', 'Flos', 'Floz', 'Fo', 'd', 'oo', 'r',
               
               'Fi', 'is', 'Nn', 'Finat', 'Nj', 'Firat', 'Nj', 'Finat', 'nj', 'Fiat', 'Ni', 'Firat', 'Ni', 'Dias', 'Floc',
                'Fina', 'BG', 'Dis', 'Firat', 'Finat', 'i', 'eae', 'Fleet', 'is', 'O', 'Finat', 'Flnat', 'Finat', 'Ni', 'as',
                'Firat', 'i', 'rae', 'Firat', 'ij', 'aia', 'apne', 'Fost', 'Vis', 'BFSB', 'Fina', 'Firat', 'Firat', 'ij', 'bis', 'Sa',
                'pe', 'Finat', 'ij', 'a', 'Fost', 'is', 'Fost', 'Dis', 'Fiat', 'ares', 'G', 'B', 'Fest', 'pares', 'AG', 'Diz',
                'Flock', 'is', 'Sead', 'Finat', 'Diz', 'Disa', 'Oueeeeee', 'Fost', 'is', 'Finat', 'ij', 'Fost', 'bis', 'Fiat', 'Seas',
                'ae', 'Fiat', 'Nj']

replace_dict_qs = {'Blinkin': 'Blinking',
                'Blink': 'Blinking',
                'Blinkir': 'Blinking',
                'Blinkit': 'Blinking',
                'Binkin': 'Blinking',
                'Bink': 'Blinking',
                'Binki': 'Blinking',
                'Blinki': 'Blinking',
                'Blinkt': 'Blinking',

                'Data': 'Data Gap',
                'Daa': 'Data Gap',
                'Data G': 'Data Gap',
                'Daa G': 'Data Gap',
                'Data C': 'Data Gap',
                'Data E': 'Data Gap',

                'Seg': 'SegB',
                'Seg E': 'SegB',
                'SegBI': 'SegB',
                'SeaB': 'SegB',
                'Seg Bli': 'SegB',
                'Seg BI': 'SegB',
                'SeqBI': 'SegB',
                'SeaBI': 'SegB',
                'SegE': 'SegB',
                'SegBli': 'SegB',
                'Seg B': 'SegB',
                'SeqB': 'SegB',
                'Seq': 'SegB',

                'Fixation': 'Fixation',
                'Fixatic': 'Fixation',
                'Fixcatio': 'Fixation',
                'Fixatio': 'Fixation',
                'Fixatior': 'Fixation',
                'Fixatic': 'Fixation',
                'Fixati': 'Fixation',
                'Fixat': 'Fixation',

                'Alian': 'Align',
                'Align I': 'Align',
                'Alion': 'Align',

                'LOK': 'OK',
                'JOK': 'OK',
                'ОК': 'OK',

                'Lidl': 'Lid',
                'Lid I': 'Lid',
                'Light S': 'Light',

                'Mode': 'Model',
                'Model I': 'Model',

                'Lov': '', 'ELO': '', '': ''
                
                
                }

oculyzer_headers = ['WAVELIGHT ALLEGRO OCULYZER 4 Maps Refractive', 
                    'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable']
pentacam_headers = ['OCULUS PENTACAM 4 Maps Refractive',
                    'OCULUS PENTACAM 4 Maps Selectable']


required_maps = ['Axial Sagittal Curvature Front', 'Elevation Front', 
                 'Corneal Thickness', 'Elevation Back']

drop_map_combos = ['Corneal Thickness, Elevation Front', 
                   'Axial Sagittal Curvature Front, Elevation Back, Elevation Front']

maps_for_4M = ['Axial Sagittal Curvature Front', 'Elevation Front', 'Corneal Thickness', 'Elevation Back']
maps_for_3M = ['Axial Sagittal Curvature Front', 'Elevation Front', 'Corneal Thickness']
maps_for_2M = ['Axial Sagittal Curvature Front', 'Corneal Thickness']

##################################################################


refractive_maps_coordinates = {
    (740, 1200): {'map1': {'row1':110, 'row2':340, 'col1':420, 'col2':650},
        'map2': {'row1':110, 'row2':340, 'col1':843, 'col2':1073},
        'map3': {'row1':405, 'row2':635, 'col1':420, 'col2':650},
        'map4': {'row1':405, 'row2':635, 'col1':843, 'col2':1073},
        'circle_loc': {'cir_x':115, 'cir_y':115, 'cir_radius':111}
    },
    (758, 1200): {'map1': {'row1':115, 'row2':387, 'col1':431, 'col2':703},
        'map2': {'row1':115, 'row2':387, 'col1':827, 'col2':1099},
        'map3': {'row1':455, 'row2':727, 'col1':431, 'col2':703},
        'map4': {'row1':455, 'row2':727, 'col1':827, 'col2':1099},
        'circle_loc': {'cir_x':136, 'cir_y':136, 'cir_radius':126}
    },
    (820, 1200): {'map1': {'row1':124, 'row2':416, 'col1':441, 'col2':733},
        'map2': {'row1':124, 'row2':416, 'col1':820, 'col2':1112},
        'map3': {'row1':493, 'row2':785, 'col1':441, 'col2':733},
        'map4': {'row1':493, 'row2':785, 'col1':820, 'col2':1112},
        'circle_loc': {'cir_x':146, 'cir_y':146, 'cir_radius':138}
    },
    (838, 1200): {'map1': {'row1':128, 'row2':424, 'col1':447, 'col2':741},
        'map2': {'row1':128, 'row2':424, 'col1':821, 'col2':1115},
        'map3': {'row1':506, 'row2':802, 'col1':447, 'col2':741},
        'map4': {'row1':506, 'row2':802,'col1':821, 'col2':1115},
        'circle_loc': {'cir_x':146, 'cir_y':148, 'cir_radius':140}
    },
    (840, 1200): {'map1': {'row1':128, 'row2':424, 'col1':447, 'col2':743},
        'map2': {'row1':128, 'row2':424, 'col1':821, 'col2':1117},
        'map3': {'row1':506, 'row2':802, 'col1':447, 'col2':743},
        'map4': {'row1':506, 'row2':802,'col1':821, 'col2':1117},
        'circle_loc': {'cir_x':148, 'cir_y':148, 'cir_radius':140}
    },
    (894, 1600): {'map1': {'row1':132, 'row2':448, 'col1':559, 'col2':875},
        'map2': {'row1':132, 'row2':448, 'col1':1117, 'col2':1433},
        'map3': {'row1':533, 'row2':849, 'col1':559, 'col2':875},
        'map4': {'row1':533, 'row2':849,'col1':1117, 'col2':1433},
        'circle_loc': {'cir_x':158, 'cir_y':158, 'cir_radius':150}
    },
    (858, 1200): {'map1': {'row1':220, 'row2':484, 'col1':433, 'col2':697},
        'map2': {'row1':220, 'row2':484, 'col1':830, 'col2':1094},
        'map3': {'row1':558, 'row2':822, 'col1':433, 'col2':697},
        'map4': {'row1':558, 'row2':822,'col1':830, 'col2':1094},
        'circle_loc': {'cir_x':132, 'cir_y':132, 'cir_radius':126}
    },
    (904, 1200): {'map1': {'row1':190, 'row2':490, 'col1':445, 'col2':741},
        'map2': {'row1':190, 'row2':490, 'col1':819, 'col2':1115},
        'map3': {'row1':568, 'row2':868, 'col1':445, 'col2':741},
        'map4': {'row1':568, 'row2':868,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':150, 'cir_radius':140}
    },
    (910, 1200): {'map1': {'row1':124, 'row2':424, 'col1':445, 'col2':741},
        'map2': {'row1':124, 'row2':424, 'col1':819, 'col2':1115},
        'map3': {'row1':500, 'row2':800, 'col1':445, 'col2':741},
        'map4': {'row1':500, 'row2':800,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':150, 'cir_radius':140}
    },
    (912, 1202): {'map1': {'row1':202, 'row2':492, 'col1':445, 'col2':741},
        'map2': {'row1':202, 'row2':492, 'col1':819, 'col2':1115},
        'map3': {'row1':578, 'row2':878, 'col1':445, 'col2':741},
        'map4': {'row1':578, 'row2':878,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':148, 'cir_y':146, 'cir_radius':142}
    },
    (940, 1200): {'map1': {'row1':230, 'row2':526, 'col1':445, 'col2':741},
        'map2': {'row1':230, 'row2':526, 'col1':819, 'col2':1115},
        'map3': {'row1':606, 'row2':902, 'col1':445, 'col2':741},
        'map4': {'row1':606, 'row2':902,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':147, 'cir_radius':140}
    }

}

selective_maps_coordinates = {
    (758, 1200):{'map1': {'row1':115, 'row2':387, 'col1':431, 'col2':703},
        'map2': {'row1':115, 'row2':387, 'col1':827, 'col2':1099},
        'map3': {'row1':455, 'row2':727, 'col1':431, 'col2':703},
        'map4': {'row1':455, 'row2':727, 'col1':827, 'col2':1099},
        'circle_loc': {'cir_x':136, 'cir_y':136, 'cir_radius':126}
    },
        
    (820, 1200):{'map1': {'row1':94, 'row2':386, 'col1':441, 'col2':733},
        'map2': {'row1':94, 'row2':386, 'col1':820, 'col2':1112},
        'map3': {'row1':483, 'row2':775, 'col1':441, 'col2':733},
        'map4': {'row1':483, 'row2':775, 'col1':820, 'col2':1112},
        'circle_loc': {'cir_x':146, 'cir_y':146, 'cir_radius':142}
    },
    (838, 1200):{'map1': {'row1':128, 'row2':424, 'col1':447, 'col2':741},
        'map2': {'row1':128, 'row2':424, 'col1':821, 'col2':1115},
        'map3': {'row1':506, 'row2':802, 'col1':447, 'col2':741},
        'map4': {'row1':506, 'row2':802,'col1':821, 'col2':1115},
        'circle_loc': {'cir_x':146, 'cir_y':148, 'cir_radius':140}
    },
    (840, 1200):{'map1': {'row1':94, 'row2':390, 'col1':447, 'col2':743},
        'map2': {'row1':94, 'row2':390, 'col1':821, 'col2':1117},
        'map3': {'row1':495, 'row2':787, 'col1':447, 'col2':743},
        'map4': {'row1':495, 'row2':787,'col1':821, 'col2':1117},
        'circle_loc': {'cir_x':148, 'cir_y':148, 'cir_radius':140}
    },
    (894, 1600):{'map1': {'row1':93, 'row2':419, 'col1':554, 'col2':880},
        'map2': {'row1':93, 'row2':419, 'col1':1112, 'col2':1438},
        'map3': {'row1':515, 'row2':841, 'col1':554, 'col2':880},
        'map4': {'row1':515, 'row2':841,'col1':1112, 'col2':1438},
        'circle_loc': {'cir_x':164, 'cir_y':163, 'cir_radius':158}
    },
        
    (858, 1200):{'map1': {'row1':220, 'row2':484, 'col1':433, 'col2':697},
        'map2': {'row1':220, 'row2':484, 'col1':830, 'col2':1094},
        'map3': {'row1':558, 'row2':793, 'col1':433, 'col2':697},
        'map4': {'row1':558, 'row2':793,'col1':830, 'col2':1094},
        'circle_loc': {'cir_x':132, 'cir_y':132, 'cir_radius':126}
    },

    (904, 1200):{'map1': {'row1':190, 'row2':490, 'col1':445, 'col2':741},
        'map2': {'row1':190, 'row2':490, 'col1':819, 'col2':1115},
        'map3': {'row1':568, 'row2':868, 'col1':445, 'col2':741},
        'map4': {'row1':568, 'row2':868,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':150, 'cir_radius':140}
    },

    (910, 1200): {'map1': {'row1':94, 'row2':394, 'col1':445, 'col2':741},
        'map2': {'row1':94, 'row2':394, 'col1':819, 'col2':1115},
        'map3': {'row1':470, 'row2':770, 'col1':445, 'col2':741},
        'map4': {'row1':470, 'row2':770,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':150, 'cir_radius':140}
    },

    (912, 1202): {'map1': {'row1':172, 'row2':462, 'col1':445, 'col2':741},
        'map2': {'row1':172, 'row2':462, 'col1':819, 'col2':1115},
        'map3': {'row1':542, 'row2':842, 'col1':445, 'col2':741},
        'map4': {'row1':542, 'row2':842,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':148, 'cir_y':146, 'cir_radius':142}
    },
        
    (940, 1200):{'map1': {'row1':230, 'row2':526, 'col1':445, 'col2':741},
        'map2': {'row1':230, 'row2':526, 'col1':819, 'col2':1115},
        'map3': {'row1':606, 'row2':902, 'col1':445, 'col2':741},
        'map4': {'row1':606, 'row2':902,'col1':819, 'col2':1115},
        'circle_loc': {'cir_x':147, 'cir_y':147, 'cir_radius':140}
    }
}

map_name_dict = {'Axial Sagittal Curvature Front':'axial_curvature', 
                 'Elevation Front':'elevation_front', 
                 'Corneal Thickness':'corneal_thickness', 
                 'Elevation Back':'elevation_back'}
map_number_dict = {'header1':'map1',
                   'header2':'map2',
                   'header3':'map3',
                   'header4':'map4'}
refractive_maps = ['OCULUS PENTACAM 4 Maps Refractive', 
                   'WAVELIGHT ALLEGRO OCULYZER 4 Maps Refractive']
selectable_maps = ['OCULUS PENTACAM 4 Maps Selectable', 
                   'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable']
map_titles = ['header1', 'header2', 'header3', 'header4']

cir_ac_3mm = {'cir_x':112, 'cir_y':112, 'cir_radius':35}
cir_ac_5mm = {'cir_x':112, 'cir_y':112, 'cir_radius':61}
cir_ac_7mm = {'cir_x':112, 'cir_y':112, 'cir_radius':85}

map_title_cols = ['header1', 'header2', 'header3', 'header4']


metric_column_rename_dict = {
    'k1_cf': 'K1_anterior',
    'k2_cf': 'K2_anterior',
    'Km_cf': 'K_mean',
    'kmax_front': 'K_max',
    'kmax_x': 'K_max_x',
    'kmax_y': 'K_max_y',
    'Astig_cf': 'Anterior_corneal_astigmatism',
    'Axis_cf': 'Axis_anterior',
    'k1_cb': 'K1_posterior',
    'k2_cb': 'K2_posterior',
    'Km_cb': 'Km_posterior',
    'Astig_cb': 'Posterior_corneal_astigmatism',
    'Axis_cb': 'Axis_posterior',
    'pachy_apex': 'Central_pachy',
    'thinnest_loc': 'Thinnest_pachy',
    'tl_x': 'Position_thin_pachy_X',
    'tl_y': 'Position_thin_pachy_Y',
    'cornea_volume': 'Corneal_vol',
    'ac_dept_int': 'ACD',

    'Rf_cf': 'Rf_anterior',
    'Rs_cf': 'Rs_anterior',
    'Rm_cf': 'Rm_anterior',
    'Rper_cf': 'Rper_anterior',
    'Rmin_cf': 'Rmin_anterior',
    
    'Rf_cb': 'Rf_posterior',
    'Rs_cb': 'Rs_posterior',
    'Rm_cb': 'Rm_posterior',
    'Rper_cb': 'Rper_posterior',
    'Rmin_cb': 'Rmin_posterior',

    'pupil_center': 'Pupil_center',
    'pc_x': 'Pupil_center_x',
    'pc_y': 'Pupil_center_y',

    'pa_x':'Central_pachy_x',
    'pa_y':'Central_pachy_y'
    
}

metric_cols = ['Rf_anterior', 'K1_anterior', 'Rs_anterior', 'K2_anterior',
                'Rm_anterior', 'K_mean', 'Axis_anterior',
                'Anterior_corneal_astigmatism', 'Rper_anterior', 'Rmin_anterior',
                'Rf_posterior', 'K1_posterior', 'Rs_posterior', 'K2_posterior',
                'Rm_posterior', 'Km_posterior', 'Axis_posterior',
                'Posterior_corneal_astigmatism', 'Rper_posterior',
                'Rmin_posterior', 'Pupil_center', 'Pupil_center_x',
                'Pupil_center_y', 'Central_pachy', 'Central_pachy_x',
                'Central_pachy_y', 'Thinnest_pachy', 'Position_thin_pachy_X',
                'Position_thin_pachy_Y', 'K_max', 'K_max_x', 'K_max_y',
                'Corneal_vol', 'chamber_vol', 'angle', 'ACD', 'pupil_dia',
                'image_name']