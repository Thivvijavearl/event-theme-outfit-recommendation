import os
import sys
import csv
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
from backend.Predict import ColorClassifier
from PIL import Image, ImageDraw, ImageFont

ALLOWED_EXT = ('.jpg', '.jpeg', '.png')
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'predictions')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'predictions.csv')


def ensure_out():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


def annotate_and_save(in_path, out_path, text):
    im = Image.open(in_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font = ImageFont.load_default()

    # small white strip at top
    h = 28
    draw.rectangle([0, 0, im.width, h], fill=(255, 255, 255))
    draw.text((6, 4), text, fill='black', font=font)
    im.save(out_path)


def gather_images(root):
    imgs = []
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ALLOWED_EXT):
                imgs.append(os.path.join(dirpath, f))
    return imgs


def main(limit=None):
    ensure_out()
    imgs = gather_images(INPUT_DIR)
    print(f'Found {len(imgs)} images under {INPUT_DIR}')
    if limit:
        imgs = imgs[:limit]

    clf = ColorClassifier()

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['filename', 'predicted_label', 'confidence'])

        for path in imgs:
            try:
                label, conf = clf.predict(path, show_image=False)
                rel = os.path.relpath(path)
                writer.writerow([rel, label, f'{conf:.4f}'])
                outname = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0] + '_pred.png')
                annotate_and_save(path, outname, f'{label} ({conf:.2%})')
            except Exception as e:
                print(f'Error on {path}: {e}')

    print(f'Done — results saved to {CSV_PATH} and annotated images to {OUT_DIR}')


if __name__ == '__main__':
    # Limit None = all images; set to an int to test quickly
    main(limit=None)
