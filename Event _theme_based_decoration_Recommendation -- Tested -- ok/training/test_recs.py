import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
from backend.MobileApi_Color import control_data

color = 'Red'
occasion = 'Wedding'

matching_colors = control_data[color]['matching_colors']
print('matching_colors', matching_colors)

def build_detailed_recs(color_list, gender):
    recs = []
    for col in color_list:
        info = control_data.get(col, {})
        color_recs = info.get('recommendations', {}).get(occasion, {}).get(gender, [])
        if color_recs:
            recs.append(color_recs[0])

    if len(recs) < 4 and color_list:
        first = color_list[0]
        extra = control_data.get(first, {}).get('recommendations', {}).get(occasion, {}).get(gender, [])[1:]
        recs += extra[: max(0, 4 - len(recs))]

    return recs

print('Men recs', build_detailed_recs(matching_colors, 'men'))
print('Women recs', build_detailed_recs(matching_colors, 'women'))
