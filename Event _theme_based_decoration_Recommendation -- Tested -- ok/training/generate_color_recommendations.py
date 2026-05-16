import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRODUCTS_CSV = os.path.join(BASE_DIR, 'data', 'new dataset', 'Wardrobe Assistant.csv')
STYLE_GUIDE_CSV = os.path.join(BASE_DIR, 'data', 'new dataset', 'recommendations.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'color_dataset_recommendations.json')

APP_OCCASIONS = [
    'Wedding',
    'Birthday',
    'Office Party',
    'Casual Outing',
    'Formal Event',
    'Beach Event',
    'Dinner Date',
    'Sports Event'
]

COLOR_CLASSES = ['Black', 'Blue', 'Green', 'Purble', 'Red', 'Red-Yellow', 'Yellow']

MATCHING_COLOR_SETS = {
    'Black': ['Gold', 'White', 'Emerald', 'Red'],
    'Blue': ['White', 'Grey', 'Beige', 'Silver'],
    'Green': ['Beige', 'Cream', 'Gold', 'Brown'],
    'Purble': ['Silver', 'White', 'Black', 'Gold'],
    'Red': ['Black', 'Gold', 'Cream', 'Navy'],
    'Red-Yellow': ['Cream', 'Black', 'Gold', 'Navy'],
    'Yellow': ['Navy', 'White', 'Olive Green', 'Grey']
}

OCCASION_MAP = {
    'wedding/formal': 'Wedding',
    'casual': 'Casual Outing',
    'party/festive': 'Birthday',
    'other': 'Casual Outing'
}

GENDER_MAP = {
    'male': 'men',
    'female': 'women'
}

SPLIT_REGEX = re.compile(r'[;,]')


def normalize_color(raw_color):
    if not isinstance(raw_color, str) or not raw_color.strip():
        return 'Blue'
    value = raw_color.strip().lower()
    if any(token in value for token in ['black', 'charcoal', 'grey', 'gray']):
        return 'Black'
    if any(token in value for token in ['purple', 'plum', 'mauve', 'violet', 'lavender']):
        return 'Purble'
    if any(token in value for token in ['red', 'maroon', 'wine', 'burgundy', 'pink', 'magenta', 'rose', 'cherry']):
        return 'Red'
    if any(token in value for token in ['yellow', 'gold', 'mustard', 'cream', 'beige', 'off white', 'ivory']):
        return 'Yellow'
    if any(token in value for token in ['orange', 'peach', 'coral', 'amber']):
        return 'Red-Yellow'
    if any(token in value for token in ['green', 'mint', 'olive', 'sage', 'emerald', 'seafoam', 'pistachio']):
        return 'Green'
    if any(token in value for token in ['blue', 'navy', 'teal', 'azure', 'sky', 'cobalt', 'indigo']):
        return 'Blue'
    # Fallback for light/neutrals and unusual names
    if any(token in value for token in ['white', 'off white', 'cream', 'beige', 'light']):
        return 'Yellow'
    return 'Blue'


def map_occasion(raw_occasion):
    if not isinstance(raw_occasion, str) or not raw_occasion.strip():
        return 'Casual Outing'
    normalized = raw_occasion.strip().lower()
    for key, mapped in OCCASION_MAP.items():
        if key in normalized:
            return mapped
    if 'formal' in normalized:
        return 'Formal Event'
    if 'wedding' in normalized:
        return 'Wedding'
    if 'party' in normalized:
        return 'Birthday'
    if 'beach' in normalized or 'summer' in normalized:
        return 'Beach Event'
    if 'sport' in normalized:
        return 'Sports Event'
    return 'Casual Outing'


def safe_string(value):
    if isinstance(value, str):
        return value.strip()
    return ''


def top_terms(series, top_n=5):
    counter = Counter()
    for value in series.dropna():
        for token in SPLIT_REGEX.split(str(value)):
            item = token.strip()
            if item:
                counter[item] += 1
    return [term for term, _ in counter.most_common(top_n)]


def format_recommendation(row):
    parts = []
    if safe_string(row.get('product_name')):
        parts.append(row['product_name'])
    if safe_string(row.get('main_category')):
        parts.append(row['main_category'])
    if safe_string(row.get('color')):
        parts.append(row['color'])
    return ' | '.join(parts)


def build_recommendation_data():
    products_df = pd.read_csv(PRODUCTS_CSV)
    style_df = pd.read_csv(STYLE_GUIDE_CSV)

    # Build global fallback recommendations
    fallback_recs = {'men': [], 'women': []}
    product_rows = []
    for _, row in products_df.iterrows():
        gender = GENDER_MAP.get(str(row.get('Gender', '')).strip().lower(), None)
        if gender is None:
            continue
        text = format_recommendation(row)
        if text:
            product_rows.append((gender, text))
            fallback_recs[gender].append(text)

    fallback_recs['men'] = list(dict.fromkeys(fallback_recs['men']))[:8]
    fallback_recs['women'] = list(dict.fromkeys(fallback_recs['women']))[:8]

    # Build recommendations grouped by normalized color and app occasion
    grouped = defaultdict(lambda: defaultdict(lambda: {'men': [], 'women': []}))
    suitable_color_counts = defaultdict(lambda: defaultdict(Counter))
    available_occasions = set()

    for _, row in products_df.iterrows():
        base_color = normalize_color(row.get('color', ''))
        occasion = map_occasion(row.get('occasion', ''))
        available_occasions.add(occasion)
        gender = GENDER_MAP.get(str(row.get('Gender', '')).strip().lower(), None)
        if gender is None:
            continue
        rec_text = format_recommendation(row)
        if not rec_text:
            continue
        if rec_text not in grouped[base_color][occasion][gender]:
            grouped[base_color][occasion][gender].append(rec_text)
        color_text = safe_string(row.get('color'))
        if color_text:
            suitable_color_counts[base_color][occasion][color_text] += 1

    all_occasions = sorted(available_occasions) if available_occasions else APP_OCCASIONS

    # Compute descriptive style guidance using body-profile dataset
    top_materials = top_terms(style_df['Recommended Materials'], top_n=5)
    top_patterns = top_terms(style_df['Recommended Patterns'], top_n=5)
    top_colors = top_terms(style_df['Recommended Clothing Colors'], top_n=5)
    top_jewelry = top_terms(style_df['Recommended Jewelry Metal'], top_n=5)

    description_template = (
        'Dataset-derived style guidance built from wardrobe recommendations and product examples. '
        'Choose flattering materials like {materials} and patterns such as {patterns} for color-smart outfits. '
        'Preferred accent tones include {colors}. '
        'Keep jewelry subtle with {jewelry}. '
        'These suggestions combine both product-level recommendations and body-style guidance.'
    )
    description_text = description_template.format(
        materials=', '.join(top_materials),
        patterns=', '.join(top_patterns),
        colors=', '.join(top_colors),
        jewelry=', '.join(top_jewelry)
    )
    dress_pattern_description = (
        'Based on the dataset, select structured or flowy fabrics and balanced patterns. '
        'For most occasions, use {materials} and {patterns} to complement the chosen color palette.'
    ).format(materials=', '.join(top_materials), patterns=', '.join(top_patterns))

    output_data = {'occasions': all_occasions}

    for base_color in COLOR_CLASSES:
        color_recs = {
            'recommendations': {},
            'description': description_text,
            'matching_colors': MATCHING_COLOR_SETS.get(base_color, []),
            'suitable_colors': {},
            'dress_pattern_description': dress_pattern_description
        }
        for occasion in all_occasions:
            men_items = grouped[base_color][occasion]['men'][:6]
            women_items = grouped[base_color][occasion]['women'][:6]
            if not men_items:
                men_items = fallback_recs['men'][:4]
            if not women_items:
                women_items = fallback_recs['women'][:4]

            top_suitable_colors = [color for color, _ in suitable_color_counts[base_color][occasion].most_common(4)]
            if not top_suitable_colors:
                top_suitable_colors = MATCHING_COLOR_SETS.get(base_color, [])[:4]

            color_recs['recommendations'][occasion] = {
                'men': men_items[:4],
                'women': women_items[:4]
            }
            color_recs['suitable_colors'][occasion] = top_suitable_colors
        output_data[base_color] = color_recs

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f'Generated recommendation JSON at: {OUTPUT_JSON}')
    print('Sample colors:', list(output_data.keys())[:8])


if __name__ == '__main__':
    build_recommendation_data()
