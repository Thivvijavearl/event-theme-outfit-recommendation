import json
try:
    with open(r'd:\Research final model tryyy\Event _theme_based_decoration_Recommendation -- Tested -- ok\data\color_dataset_recommendations.json', 'r') as f:
        json.load(f)
    print('JSON is valid')
except Exception as e:
    print('Error:', e)