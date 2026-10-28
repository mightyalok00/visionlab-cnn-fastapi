from pathlib import Path
from PIL import Image
from tensorflow import keras
from app.model import CLASS_NAMES

def main():
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    target_classes = ["T-shirt/top", "Trouser", "Dress", "Coat", "Sandal", "Sneaker", "Bag", "Ankle boot"]
    selected = {}
    for img, label in zip(x_test, y_test):
        cname = CLASS_NAMES[label]
        if cname in target_classes and cname not in selected:
            selected[cname] = img
        if len(selected) == len(target_classes):
            break

    for cname, arr in selected.items():
        safe_name = cname.replace("/", "_").lower()
        im = Image.fromarray(arr)
        im.resize((112, 112), Image.Resampling.NEAREST).save(samples_dir / f"{safe_name}_dark_bg.png")
        im_inv = Image.fromarray(255 - arr)
        im_inv.resize((112, 112), Image.Resampling.NEAREST).save(samples_dir / f"{safe_name}_light_bg.png")

    count = len(list(samples_dir.glob("*.png")))
    print(f"Generated {count} sample images in samples/")

if __name__ == "__main__":
    main()
