"""
Railway Crack Detection — YOLOv8 Training Script
================================================

A simple, streamlined script for training and evaluating a YOLOv8 model.
For dataset analysis, use `dataset_analyzer.py` instead.
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Railway Crack Detection — YOLOv8 Training")
    
    # Data & Model
    parser.add_argument("--data-yaml", type=str, default="dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    
    # Training Hyperparameters
    parser.add_argument("--image-size", type=int, default=640, help="Training input size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', None=auto)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training")
    parser.add_argument("--half", action="store_true", help="Enable mixed-precision (FP16)")
    
    # Run tracking
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--run-suffix", type=str, default=None, help="Suffix for run name")
    parser.add_argument("--skip-test", action="store_true", help="Skip test-set evaluation")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    
    # Augmentation
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--hsv-h", type=float, default=0.01)
    parser.add_argument("--hsv-s", type=float, default=0.2)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    
    return parser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def run_preflight(data_yaml_path: str):
    """Sanity check before launching training."""
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {yaml_path.resolve()}")
        print("Please verify the path or run 'python dataset_analyzer.py --check'")
        sys.exit(1)
        
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Error reading {yaml_path.name}: {e}")
        sys.exit(1)
        
    # Check if the directories exist
    parent = yaml_path.parent
    train_dir = config.get("train", "train/images")
    if not (parent / train_dir).exists() and not Path(train_dir).exists():
         print(f"[ERROR] Train images directory '{train_dir}' not found.")
         print("Please run 'python dataset_analyzer.py --check' for full validation.")
         sys.exit(1)

def extract_metrics(metrics_obj, split_name: str) -> dict:
    """Extract P, R, mAP and calculate F1."""
    try:
        p = float(metrics_obj.box.mp)
        r = float(metrics_obj.box.mr)
        map50 = float(metrics_obj.box.map50)
        map50_95 = float(metrics_obj.box.map)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        
        print(f"\n[STATS] {split_name} Results:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  mAP@50:    {map50:.4f}")
        print(f"  mAP@50-95: {map50_95:.4f}")
        
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": map50,
            "map50_95": map50_95
        }
    except Exception as e:
        print(f"[WARNING] Could not extract metrics: {e}")
        return {}

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    run_preflight(args.data_yaml)
    set_seed(args.seed)
    
    # Auto-generate unique run name
    run_name = f"{Path(args.model).stem}_{args.image_size}"
    if args.rect: run_name += "_rect"
    if args.half: run_name += "_half"
    if args.run_suffix: run_name += f"_{args.run_suffix}"
    
    run_dir = Path(args.project) / run_name
    if run_dir.exists():
        run_name += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(args.project) / run_name
    
    print(f"\n[START] Starting Training: {run_name}")
    
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=args.data_yaml,
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=run_name,
        exist_ok=False,  # Directory created by Ultralytics
        patience=args.patience,
        rect=args.rect,
        half=args.half,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        save_period=args.save_period,
    )
    
    # The actual run_dir might have been altered by Ultralytics if exist_ok=False and the dir somehow existed.
    actual_run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else run_dir
    best_weights = actual_run_dir / "weights" / "best.pt"
    
    if not best_weights.exists():
        print(f"[ERROR] Training failed to produce best.pt at {best_weights}")
        sys.exit(1)
        
    print(f"\n[SUCCESS] Training complete. Best model: {best_weights}")
    
    best_model = YOLO(str(best_weights))
    summary = vars(args).copy()
    summary["timestamp"] = datetime.datetime.now().isoformat()
    
    # Validate
    print("\n[STATS] Validating...")
    val_metrics = best_model.val(data=args.data_yaml, split="val", imgsz=args.image_size, device=args.device)
    summary["val_metrics"] = extract_metrics(val_metrics, "Validation")
    
    # Test
    if not args.skip_test:
        print("\n[TEST] Testing on held-out set...")
        test_metrics = best_model.val(data=args.data_yaml, split="test", imgsz=args.image_size, device=args.device)
        summary["test_metrics"] = extract_metrics(test_metrics, "Test")
    else:
        print("\n[SKIP] Skipped test set evaluation (--skip-test).")

    # Save summary
    summary_path = actual_run_dir / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Experiment summary saved to {summary_path}")

if __name__ == "__main__":
    main()