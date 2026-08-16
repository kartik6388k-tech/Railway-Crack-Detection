from __future__ import annotations

import os

# On Windows, PyTorch/Ultralytics and OpenCV are often each statically linked
# against Intel's OpenMP runtime (libiomp5md.dll). When both get loaded in the
# same process, the SECOND one to initialize can trigger "OMP: Error #15:
# Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# That is a native-level abort, not a Python exception — it kills the whole
# process mid-request, so Flask never finishes writing the JSON response and
# the browser gets an empty body ("Unexpected end of JSON input"). Setting
# this before torch/ultralytics is imported works around it. This must run
# before the `from ultralytics import YOLO` line below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    print(f"[ERROR] ultralytics package not found: {exc}")
    print("Install it with: pip install ultralytics")
    raise SystemExit(1)


DEFAULT_WEIGHTS = Path(
    r"E:\rail-wheel-crack-detection\pretrained weights\best.pt"
)
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_CONFIDENCE = 0.10
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 960
DEFAULT_MAX_DETECTIONS = 100
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Railway Crack/Gap Detection — YOLOv8 + OpenCV"
    )
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained YOLO .pt weights")
    parser.add_argument("--output", default=None, help="Output image path. If omitted, results/<name>_detected.<ext> is used.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold (default: 0.10)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Inference image size (default: 960)")
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DETECTIONS, help="Maximum number of detections per image (default: 100)")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 or cpu. Default: auto")
    parser.add_argument("--show", action="store_true", help="Display the annotated result using OpenCV")
    return parser


def validate_arguments(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = Path(args.input)
    weights_path = Path(args.weights)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found:\n{input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file:\n{input_path}")
    if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found:\n{weights_path}\nUse --weights to specify the correct best.pt path."
        )
    if weights_path.suffix.lower() != ".pt":
        raise ValueError(f"Weights file must be a .pt file, got: {weights_path.suffix}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0.")
    if args.imgsz % 32 != 0:
        print(f"[WARNING] --imgsz {args.imgsz} is not a multiple of 32; YOLO will round it internally.")
    if args.max_det <= 0:
        raise ValueError("--max-det must be greater than 0.")

    return input_path, weights_path


def load_model(weights_path: Path) -> YOLO:
    try:
        model = YOLO(str(weights_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model from {weights_path}: {exc}") from exc
    return model


def detect_image(
    model: YOLO,
    image_path: Path,
    confidence: float,
    iou: float,
    image_size: int,
    max_detections: int,
    device: str | None,
):
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"OpenCV could not read the image:\n{image_path}")

    original_height, original_width = original.shape[:2]

    print(f"[detect] starting model.predict on {image_path}", flush=True)
    try:
        results = model.predict(
            source=str(image_path),
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            max_det=max_detections,
            device=device,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc
    print("[detect] model.predict finished", flush=True)

    if not results:
        raise RuntimeError("YOLO returned no result.")

    result = results[0]
    annotated = result.plot()
    print("[detect] result.plot() finished", flush=True)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for box, score, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width - 1))
            y2 = max(0, min(y2, original_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = names.get(int(class_id), str(class_id))
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(score),
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                }
            )

    detections.sort(key=lambda detection: detection["confidence"], reverse=True)
    return original, annotated, detections


def save_result(annotated_image, input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_detected{input_path.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), annotated_image)
    if not success:
        raise RuntimeError(f"Could not save output image:\n{output_path}")

    return output_path


def print_detection_results(detections: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("DETECTION RESULT")
    print("=" * 60)

    if not detections:
        print("\nNo defect detected above the confidence threshold.")
        return

    print(f"\nTotal detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        bbox = detection["bbox"]
        print(
            f"\nDetection {index}:"
            f"\n  Class       : {detection['class_name']}"
            f"\n  Confidence  : {detection['confidence']:.4f}"
            f"\n  Bounding box:"
            f"\n    x1 = {bbox['x1']}"
            f"\n    y1 = {bbox['y1']}"
            f"\n    x2 = {bbox['x2']}"
            f"\n    y2 = {bbox['y2']}"
        )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    input_path, weights_path = validate_arguments(args)

    print("=" * 60)
    print("RAILWAY CRACK / GAP DETECTION")
    print("=" * 60)
    print(f"\nWeights       : {weights_path}")
    print(f"Input         : {input_path}")
    print(f"Confidence    : {args.conf}")
    print(f"IoU           : {args.iou}")
    print(f"Image size    : {args.imgsz}")
    print(f"Max detections: {args.max_det}")

    print("\nLoading model...")
    model = load_model(weights_path)
    print(f"Model classes: {model.names}")

    print("\nRunning detection...")
    _, annotated, detections = detect_image(
        model=model,
        image_path=input_path,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        max_detections=args.max_det,
        device=args.device,
    )

    output_path = save_result(
        annotated_image=annotated,
        input_path=input_path,
        output_arg=args.output,
    )

    print_detection_results(detections)
    print(f"\nAnnotated image saved to:\n{output_path}")

    if args.show:
        try:
            cv2.imshow("Railway Crack Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"[WARNING] Could not display image window: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\n[ERROR] {error}")
        raise SystemExit(1)
    except Exception as error:
        print(f"\n[UNEXPECTED ERROR] {error}")
        raise SystemExit(1)