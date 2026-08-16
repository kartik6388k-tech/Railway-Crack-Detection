"""
Dataset Analyzer for Railway Crack Detection
============================================

Modes:
    --check      Validate environment, data.yaml, dataset structure, image
                 readability, and label format. Checks for missing, empty,
                 and invalid labels.
    --analyze    Full dataset analysis: image dimensions, aspect ratios,
                 bounding-box pixel sizes (area, ratio), resolution 
                 recommendation, annotation visualization.

Duplicate checking:
    --check-duplicates exact    (SHA-256)
    --check-duplicates phash    (Perceptual hash, requires imagehash)
"""

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import yaml

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ["train", "valid", "test"]

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Railway Crack Detection — Dataset Analyzer")
    
    parser.add_argument("--check", action="store_true", help="Validate dataset structure and labels")
    parser.add_argument("--analyze", action="store_true", help="Full dataset analysis")
    parser.add_argument("--check-duplicates", type=str, default="none", choices=["none", "exact", "phash"])
    parser.add_argument("--phash-distance", type=int, default=4)
    
    parser.add_argument("--data-yaml", type=str, default="dataset/data.yaml")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--image-size", type=int, default=640, help="Target training resolution for recommendations")
    
    return parser

def get_image_files(directory: Path) -> List[Path]:
    if not directory.is_dir(): return []
    return sorted(f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS)

def read_label_lines(label_path: Path) -> List[str]:
    try:
        return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []

def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.split()
    if len(parts) != 5: return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None

def read_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    img = cv2.imread(str(image_path))
    if img is None: return None
    h, w = img.shape[:2]
    if w <= 0 or h <= 0: return None
    return w, h

def percentile_stats(values: List[float]) -> Dict[str, float]:
    if not values: return {}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "P5": float(np.percentile(arr, 5)),
        "P10": float(np.percentile(arr, 10)),
        "P25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "P75": float(np.percentile(arr, 75)),
        "P90": float(np.percentile(arr, 90)),
        "P95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

def format_stats_line(label: str, stats: Dict[str, float], unit: str = "px") -> str:
    if not stats: return f"  {label}: no data"
    return (
        f"  {label:>20s}:  min={stats['min']:7.1f}{unit}  P5={stats['P5']:7.1f}  "
        f"median={stats['median']:7.1f}  P95={stats['P95']:7.1f}  max={stats['max']:7.1f}{unit}"
    )

# ============================================================
# CHECK MODE
# ============================================================

def run_check(args: argparse.Namespace) -> bool:
    print("\n" + "=" * 70 + "\nCHECK MODE — Environment & Dataset Validation\n" + "=" * 70)
    errors, warnings = [], []
    
    print(f"\n  Python: {sys.version.split()[0]}")
    try: import torch; print(f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except: warnings.append("PyTorch not installed")
    try: import ultralytics; print(f"  Ultralytics: {ultralytics.__version__}")
    except: warnings.append("Ultralytics not installed")

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        errors.append(f"data.yaml not found: {data_yaml}")
        nc, names = 1, []
    else:
        with open(data_yaml, "r", encoding="utf-8") as f: data_config = yaml.safe_load(f)
        nc, names = data_config.get("nc"), data_config.get("names")
        print(f"  data.yaml: nc={nc}, names={names}")
        if nc is None or names is None: errors.append("data.yaml missing 'nc' or 'names'")
    
    class_count = nc if nc is not None else 1
    dataset_root = Path(args.dataset_root)

    if not dataset_root.is_dir():
        errors.append(f"Dataset root not found: {dataset_root}")
    else:
        for split in SPLITS:
            img_dir, lbl_dir = dataset_root / split / "images", dataset_root / split / "labels"
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                errors.append(f"Missing {split} directories"); continue
                
            img_files = get_image_files(img_dir)
            lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")}
            img_stems = {f.stem for f in img_files}
            
            print(f"  --- {split} --- \n  Images: {len(img_files)} | Labels: {len(lbl_files)}")

            # Two-way correspondence
            for img_path in img_files:
                if img_path.stem not in lbl_files: 
                    errors.append(f"[{split}] Missing label for {img_path.name}")
                if read_image_dimensions(img_path) is None: 
                    errors.append(f"[{split}] Unreadable image {img_path.name}")
                    
            for stem, lbl_path in lbl_files.items():
                if stem not in img_stems:
                    errors.append(f"[{split}] Orphaned label (no image): {lbl_path.name}")
                
                # Check label validity and zero-size boxes
                for line_num, line in enumerate(read_label_lines(lbl_path), start=1):
                    parsed = parse_yolo_line(line)
                    if not parsed: 
                        errors.append(f"[{split}] {lbl_path.name} line {line_num}: invalid format")
                    else:
                        c, xc, yc, w, h = parsed
                        if not (0 <= c < class_count): 
                            errors.append(f"[{split}] {lbl_path.name} line {line_num}: bad class")
                        if any(not (0.0 <= v <= 1.0) for v in (xc, yc, w, h)): 
                            errors.append(f"[{split}] {lbl_path.name} line {line_num}: coords out of bounds")
                        if w <= 0 or h <= 0:
                            errors.append(f"[{split}] {lbl_path.name} line {line_num}: zero-width or zero-height box")

    if errors:
        print("\n  ERRORS:"); [print(f"    ✗ {e}") for e in errors[:20]]
        if len(errors) > 20: print(f"    ... and {len(errors) - 20} more.")
        return False
    print("\nCHECK PASSED\n")
    return True

# ============================================================
# ANALYZE MODE
# ============================================================

def run_analyze(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70 + "\nANALYZE MODE — Dataset Analysis\n" + "=" * 70)
    dataset_root = Path(args.dataset_root)
    
    with open(args.data_yaml, "r", encoding="utf-8") as f: class_names = yaml.safe_load(f).get("names", [])
    all_split_data = {}

    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.is_dir(): continue
        
        split_data = {
            "dims": [], 
            "boxes": [], 
            "annotated": 0, 
            "empty": 0, 
            "invalid": 0, 
            "missing": 0,
            "boundary_issues": []
        }
        
        for img_path in get_image_files(images_dir):
            dims = read_image_dimensions(img_path)
            if not dims: continue
            iw, ih = dims
            split_data["dims"].append((img_path, iw, ih))
            
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                split_data["missing"] += 1
                continue
                
            lines = read_label_lines(lbl_path)
            
            if not lines:
                split_data["empty"] += 1
            else:
                valid_boxes = False
                has_invalid = False
                for line in lines:
                    parsed = parse_yolo_line(line)
                    if parsed:
                        c, xc, yc, w, h = parsed
                        if w > 0 and h > 0:
                            valid_boxes = True
                            split_data["boxes"].append((img_path, c, xc, yc, w, h, iw, ih))
                            
                            # Boundary check
                            x1 = (xc - w / 2) * iw
                            y1 = (yc - h / 2) * ih
                            x2 = (xc + w / 2) * iw
                            y2 = (yc + h / 2) * ih
                            if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
                                split_data["boundary_issues"].append((lbl_path.name, x1, y1, x2, y2, iw, ih))
                        else:
                            has_invalid = True
                    else:
                        has_invalid = True
                        
                if valid_boxes and not has_invalid: 
                    split_data["annotated"] += 1
                elif has_invalid: 
                    split_data["invalid"] += 1
                
        all_split_data[split] = split_data
        _report_split_stats(split, split_data)
        
    _report_recommendations(all_split_data, args.image_size)
    if args.check_duplicates != "none": _check_duplicates(all_split_data, args)
    _visualize(all_split_data, dataset_root, class_names)

def _report_split_stats(split: str, data: dict):
    print(f"\n--- {split.upper()} ---")
    dims = data["dims"]
    if not dims: return
    
    ws, hs = [w for _, w, _ in dims], [h for _, _, h in dims]
    ratios = [w/h for w, h in zip(ws, hs)]
    landscape = sum(1 for r in ratios if r > 1.05)
    portrait = sum(1 for r in ratios if r < 0.95)
    square = len(ratios) - landscape - portrait
    
    res_counter = Counter((w, h) for _, w, h in dims)
    
    print(f"  Images ({len(dims)}):")
    print(f"    Width range:  {min(ws)} - {max(ws)} (mean {np.mean(ws):.0f})")
    print(f"    Height range: {min(hs)} - {max(hs)} (mean {np.mean(hs):.0f})")
    print(f"    Aspect ratio: Landscape={landscape}, Portrait={portrait}, Square={square} | Range: {min(ratios):.2f}-{max(ratios):.2f}")
    print(f"    Top resolutions: {', '.join(f'{w}x{h} ({c})' for (w,h), c in res_counter.most_common(5))}")
    
    print(f"\n  Label Status:")
    print(f"    Annotated: {data['annotated']}")
    print(f"    Empty (Negative): {data['empty']} ({(data['empty']/len(dims)*100):.1f}%)")
    if data['missing']: print(f"    [WARNING] Missing labels: {data['missing']}")
    if data['invalid']: print(f"    [WARNING] Invalid labels: {data['invalid']}")
    if data['boundary_issues']: print(f"    [WARNING] Boundary issues: {len(data['boundary_issues'])} boxes fall outside image pixel bounds.")
    
    boxes = data["boxes"]
    if boxes:
        box_ws = [bw * iw for _, _, _, _, bw, _, iw, _ in boxes]
        box_hs = [bh * ih for _, _, _, _, _, bh, _, ih in boxes]
        box_areas = [w * h for w, h in zip(box_ws, box_hs)]
        box_pcts = [(a / (iw*ih)) * 100 for a, (_, _, _, _, _, _, iw, ih) in zip(box_areas, boxes)]
        box_ratios = [w / h if h > 0 else 0 for w, h in zip(box_ws, box_hs)]
        
        print("\n  Bounding Box Pixel Stats:")
        print(format_stats_line("Box width", percentile_stats(box_ws)))
        print(format_stats_line("Box height", percentile_stats(box_hs)))
        print(format_stats_line("Box area", percentile_stats(box_areas), unit="px²"))
        print(format_stats_line("Area % of image", percentile_stats(box_pcts), unit="%"))
        print(format_stats_line("Box aspect ratio", percentile_stats(box_ratios), unit=""))

def _report_recommendations(all_split_data: dict, current_imgsz: int):
    all_w = [bw * iw for d in all_split_data.values() for _, _, _, _, bw, _, iw, _ in d["boxes"]]
    all_iw = [iw for d in all_split_data.values() for _, iw, _ in d["dims"]]
    if not all_w: return
    
    med_iw = np.median(all_iw)
    p5_w = np.percentile(all_w, 5)
    print(f"\nRESOLUTION ESTIMATES (Informational)")
    print(f"Based on a global P5 box width of {p5_w:.1f}px and median original width of {med_iw:.0f}px.")
    print("Note: Differing aspect ratios and letterboxing will affect final training sizes.")
    for imgsz in [512, 640, 768, 960]:
        eff_p5 = p5_w * (imgsz / med_iw)
        print(f"  imgsz={imgsz:<4d} -> estimated effective P5 width = {eff_p5:.1f}px {'(Warning: potentially too small)' if eff_p5 < 10 else ''}")

def _check_duplicates(all_split_data: dict, args):
    print(f"\nDUPLICATE CHECK ({args.check_duplicates})")
    
    if args.check_duplicates == "exact":
        hashes = defaultdict(list)
        for split, data in all_split_data.items():
            for img_path, _, _ in data["dims"]:
                hashes[hashlib.sha256(img_path.read_bytes()).hexdigest()].append(f"{split}/{img_path.name}")
                
        dupes = {h: f for h, f in hashes.items() if len(f) > 1}
        for h, f in list(dupes.items())[:5]:
            print(f"  Duplicate set: {', '.join(f)}")
        if not dupes: print("  No exact duplicates found.")
        
    elif args.check_duplicates == "phash":
        try:
            import imagehash
            from PIL import Image as PILImage
        except ImportError:
            print("  [WARNING] 'imagehash' and 'Pillow' are required for pHash. Run: pip install imagehash Pillow")
            return
            
        print(f"  Calculating perceptual hashes (distance <= {args.phash_distance})...")
        phash_map = []
        for split, data in all_split_data.items():
            for img_path, _, _ in data["dims"]:
                try:
                    h = imagehash.phash(PILImage.open(str(img_path)))
                    phash_map.append((split, img_path.name, h))
                except Exception: pass
                
        near_dupes = []
        for i in range(len(phash_map)):
            for j in range(i + 1, len(phash_map)):
                s1, n1, h1 = phash_map[i]
                s2, n2, h2 = phash_map[j]
                dist = h1 - h2
                if 0 < dist <= args.phash_distance:
                    near_dupes.append((dist, s1, n1, s2, n2))
                    
        near_dupes.sort(key=lambda x: x[0])
        for dist, s1, n1, s2, n2 in near_dupes[:10]:
            print(f"  Dist {dist}: {s1}/{n1} <-> {s2}/{n2}")
        if not near_dupes: print("  No near-duplicates found.")

def _visualize(all_split_data: dict, root: Path, class_names: list):
    out_dir = Path("runs/analysis/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    
    for split, data in all_split_data.items():
        if not data["dims"]: continue
        
        # Build lookup for boxes
        boxes_by_path = defaultdict(list)
        for b in data["boxes"]:
            boxes_by_path[b[0]].append(b)
            
        # Categorize images
        landscape, portrait, negative = [], [], []
        for img_path, iw, ih in data["dims"]:
            if len(boxes_by_path[img_path]) == 0:
                negative.append((img_path, iw, ih))
            elif iw / ih > 1.05: landscape.append((img_path, iw, ih))
            elif iw / ih < 0.95: portrait.append((img_path, iw, ih))
            
        by_area = sorted(data["dims"], key=lambda x: x[1]*x[2])
        small = by_area[:5]
        large = by_area[-5:]
        
        samples = {
            "landscape": landscape[:2],
            "portrait": portrait[:2],
            "small": small[:2],
            "large": large[:2],
            "negative": negative[:2]
        }
        
        for category, imgs in samples.items():
            for img_path, iw, ih in imgs:
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                # Draw boxes
                for _, cid, xc, yc, bw, bh, _, _ in boxes_by_path[img_path]:
                    x1 = int((xc - bw / 2) * iw)
                    y1 = int((yc - bh / 2) * ih)
                    x2 = int((xc + bw / 2) * iw)
                    y2 = int((yc + bh / 2) * ih)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                    cv2.putText(img, f"{label}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Overlay category
                cv2.putText(img, f"{category} ({iw}x{ih})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.imwrite(str(out_dir / f"{split}_{category}_{img_path.name}"), img)
                count += 1
                
    print(f"\nSaved {count} representative visualization samples (with drawn boxes) to {out_dir}")

def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.check: run_check(args)
    if args.analyze: run_analyze(args)
    if not args.check and not args.analyze:
        parser.print_help()

if __name__ == "__main__":
    main()
