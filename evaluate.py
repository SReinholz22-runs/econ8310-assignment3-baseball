"""
evaluate.py — Load saved weights and run baseball detection inference
No training required. Loads model.pth and runs on a video or image folder.

Usage:
    # Run on a single video file
    python evaluate.py --source data/dusty_1.mov

    # Run on a folder of videos
    python evaluate.py --source data/

    # Save annotated output frames to a folder
    python evaluate.py --source data/dusty_1.mov --output_dir results/

    # Adjust confidence threshold
    python evaluate.py --source data/dusty_1.mov --threshold 0.7
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ── Model loading ───────────────────────────────────────────────────────────────

def load_model(weights_path, num_classes=2, device=None):
    """
    Reconstruct the Faster R-CNN architecture and load saved weights.

    Args:
        weights_path (str): Path to saved .pth file (e.g. 'weights/model.pth').
        num_classes  (int): Must match what was used during training (default: 2).
        device       (torch.device): CPU or CUDA. Auto-detected if None.

    Returns:
        model (nn.Module): Model in eval mode, ready for inference.
        device (torch.device): The device the model is on.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Loading model on: {device}")

    # Rebuild architecture without pretrained weights (we load our own)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load saved weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found at '{weights_path}'. "
            "Run train.py first to generate model weights."
        )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Weights loaded from: {weights_path}")
    return model, device


# ── Inference ───────────────────────────────────────────────────────────────────

def predict_frame(model, frame_bgr, device, threshold=0.5):
    """
    Run inference on a single BGR frame (as returned by cv2).

    Args:
        model      : Loaded Faster R-CNN model in eval mode.
        frame_bgr  : numpy array (H, W, 3) in BGR format.
        device     : torch.device
        threshold  : Confidence threshold — boxes below this are discarded.

    Returns:
        boxes  : list of [xtl, ytl, xbr, ybr] in pixel coordinates
        scores : list of float confidence scores
    """
    # Convert BGR -> RGB -> float tensor [0, 1]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        predictions = model(tensor)

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # Filter by confidence threshold and class label (1 = baseball)
    keep = [
        i for i, (s, l) in enumerate(zip(scores, labels))
        if s >= threshold and l == 1
    ]

    return [boxes[i].tolist() for i in keep], [float(scores[i]) for i in keep]


def draw_boxes(frame_bgr, boxes, scores):
    """Draw bounding boxes and scores onto a frame. Returns annotated frame."""
    annotated = frame_bgr.copy()
    for box, score in zip(boxes, scores):
        xtl, ytl, xbr, ybr = [int(v) for v in box]
        cv2.rectangle(annotated, (xtl, ytl), (xbr, ybr), color=(0, 255, 0), thickness=2)
        label = f"baseball {score:.2f}"
        cv2.putText(
            annotated, label, (xtl, ytl - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), thickness=2
        )
    return annotated


# ── Video processing ─────────────────────────────────────────────────────────────

def process_video(model, video_path, device, threshold=0.5, output_dir=None):
    """
    Run inference on every frame of a video.

    Args:
        model      : Loaded model.
        video_path : Path to .mov/.mp4 file.
        device     : torch.device
        threshold  : Confidence threshold.
        output_dir : If set, saves annotated frames as images here.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem
    print(f"\nProcessing: {video_path} ({total_frames} frames)")

    if output_dir:
        save_dir = Path(output_dir) / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    detections_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores = predict_frame(model, frame, device, threshold)

        if boxes:
            print(f"  Frame {frame_idx:04d}: {len(boxes)} detection(s) — scores: {[f'{s:.2f}' for s in scores]}")
            for box, score in zip(boxes, scores):
                detections_log.append({
                    "frame": frame_idx,
                    "box": box,
                    "score": score,
                })

        if output_dir:
            annotated = draw_boxes(frame, boxes, scores)
            out_path = save_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), annotated)

        frame_idx += 1

    cap.release()
    print(f"  Done. {len(detections_log)} total detections across {frame_idx} frames.")
    return detections_log


# ── Entry point ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseball detector on video(s)")
    parser.add_argument(
        "--weights", type=str, default="weights/model.pth",
        help="Path to saved model weights (default: weights/model.pth)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to a single video file OR a folder of videos"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence score threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="If set, saves annotated frames here (optional)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load model once
    model, device = load_model(args.weights)

    source = Path(args.source)
    video_extensions = {".mov", ".mp4", ".avi"}

    if source.is_file() and source.suffix.lower() in video_extensions:
        # Single video
        process_video(model, source, device, args.threshold, args.output_dir)

    elif source.is_dir():
        # All videos in folder
        videos = [f for f in source.iterdir() if f.suffix.lower() in video_extensions]
        if not videos:
            print(f"No video files found in '{source}'.")
        for video_path in sorted(videos):
            process_video(model, video_path, device, args.threshold, args.output_dir)

    else:
        print(f"[!] '{source}' is not a valid video file or directory.")
