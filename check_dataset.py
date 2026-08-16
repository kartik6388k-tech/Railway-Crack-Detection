from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"E:\rail-wheel-crack-detection\dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLITS = ["train", "valid", "test"]

# Expected number of classes.
# Change this if your dataset contains more than one class.
NUM_CLASSES = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all image files in a folder."""
    if not folder.exists():
        return []

    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def check_label_file(label_path):
    """
    Check a YOLO label file.

    Expected format:
    class_id x_center y_center width height
    """

    result = {
        "empty": False,
        "valid": True,
        "boxes": 0,
        "classes": [],
        "errors": []
    }

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Could not read file: {e}")
        return result

    # Empty label = no objects
    if not text:
        result["empty"] = True
        return result

    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # YOLO requires exactly 5 values
        if len(parts) != 5:
            result["valid"] = False
            result["errors"].append(
                f"Line {line_number}: expected 5 values, found {len(parts)}"
            )
            continue

        # Convert values to numbers
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            result["valid"] = False
            result["errors"].append(
                f"Line {line_number}: contains non-numeric values"
            )
            continue

        # Check class ID
        if class_id < 0 or class_id >= NUM_CLASSES:
            result["valid"] = False
            result["errors"].append(
                f"Line {line_number}: invalid class ID {class_id}"
            )

        # Check normalized coordinates
        values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        }

        for name, value in values.items():
            if not 0 <= value <= 1:
                result["valid"] = False
                result["errors"].append(
                    f"Line {line_number}: {name}={value} is outside [0, 1]"
                )

        # Width and height cannot be zero
        if width <= 0:
            result["valid"] = False
            result["errors"].append(
                f"Line {line_number}: width must be > 0"
            )

        if height <= 0:
            result["valid"] = False
            result["errors"].append(
                f"Line {line_number}: height must be > 0"
            )

        result["boxes"] += 1
        result["classes"].append(class_id)

    return result


# ============================================================
# CHECK ONE SPLIT
# ============================================================

def check_split(split):
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    if not images_dir.exists():
        print(f"[ERROR] Missing directory: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"[ERROR] Missing directory: {labels_dir}")
        return

    images = get_images(images_dir)

    labels = list(labels_dir.glob("*.txt"))

    image_names = {image.stem for image in images}
    label_names = {label.stem for label in labels}

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    print(f"Images              : {len(images)}")
    print(f"Label files         : {len(labels)}")

    # --------------------------------------------------------
    # Missing labels
    # --------------------------------------------------------

    images_without_labels = image_names - label_names

    print(f"Images without label: {len(images_without_labels)}")

    if images_without_labels:
        print("\nImages without labels:")

        for name in sorted(images_without_labels)[:20]:
            print(f"  - {name}")

        if len(images_without_labels) > 20:
            print(f"  ... and {len(images_without_labels) - 20} more")

    # --------------------------------------------------------
    # Labels without images
    # --------------------------------------------------------

    labels_without_images = label_names - image_names

    print(f"Labels without image: {len(labels_without_images)}")

    if labels_without_images:
        print("\nLabels without corresponding images:")

        for name in sorted(labels_without_images)[:20]:
            print(f"  - {name}")

        if len(labels_without_images) > 20:
            print(f"  ... and {len(labels_without_images) - 20} more")

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    empty_labels = []
    invalid_labels = []
    valid_labels = []

    class_counter = Counter()
    total_boxes = 0

    for label_path in labels:

        result = check_label_file(label_path)

        total_boxes += result["boxes"]

        for class_id in result["classes"]:
            class_counter[class_id] += 1

        if result["empty"]:
            empty_labels.append(label_path)

        elif not result["valid"]:
            invalid_labels.append((label_path, result["errors"]))

        else:
            valid_labels.append(label_path)

    # --------------------------------------------------------
    # Label statistics
    # --------------------------------------------------------

    print(f"\nValid labels        : {len(valid_labels)}")
    print(f"Empty labels        : {len(empty_labels)}")
    print(f"Invalid labels      : {len(invalid_labels)}")
    print(f"Total bounding boxes: {total_boxes}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    if class_counter:
        for class_id, count in sorted(class_counter.items()):
            print(f"  Class {class_id}: {count} boxes")
    else:
        print("  No bounding boxes found.")

    # --------------------------------------------------------
    # Empty label files
    # --------------------------------------------------------

    if empty_labels:
        print("\nEmpty label files:")

        for label in empty_labels[:20]:
            print(f"  - {label.name}")

        if len(empty_labels) > 20:
            print(f"  ... and {len(empty_labels) - 20} more")

    # --------------------------------------------------------
    # Invalid label files
    # --------------------------------------------------------

    if invalid_labels:

        print("\nINVALID LABEL FILES:")

        for label_path, errors in invalid_labels[:20]:

            print(f"\n  {label_path.name}")

            for error in errors:
                print(f"    -> {error}")

        if len(invalid_labels) > 20:
            print(f"\n  ... and {len(invalid_labels) - 20} more")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAIL-WHEEL CRACK DETECTION DATASET CHECK")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found:")
        print(f"       {DATASET_DIR.resolve()}")
        print("\nMake sure check_dataset.py is in your project root.")
        return

    for split in SPLITS:
        check_split(split)

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()