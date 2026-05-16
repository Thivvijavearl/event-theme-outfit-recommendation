import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
from backend.Predict import ColorClassifier
from PIL import Image, ImageDraw, ImageFont


def main():
    test_image = os.path.join(BASE_DIR, 'data', 'samples', 'red_001 (97).jpg')
    out_image = os.path.join(os.path.dirname(__file__), 'prediction_output.png')

    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return

    clf = ColorClassifier()
    # Use the classifier's built-in matplotlib display (old behavior)
    class_name, confidence = clf.predict(test_image, show_image=True)
    print(f"Prediction: {class_name} ({confidence:.2%})")

    # Also save an annotated copy for records
    im = Image.open(test_image).convert("RGB")
    draw = ImageDraw.Draw(im)
    text = f"{class_name} ({confidence:.2%})"
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    rect_height = 32
    draw.rectangle([0, 0, im.width, rect_height], fill=(255, 255, 255))
    draw.text((8, 6), text, fill="black", font=font)
    im.save(out_image)
    print(f"Saved annotated image to {out_image}")

if __name__ == '__main__':
    main()
