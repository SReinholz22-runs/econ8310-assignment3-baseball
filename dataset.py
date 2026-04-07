
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def parse_cvat_xml(xml_path):
    """
    Parse a CVAT XML annotation file.
    Returns a dict mapping frame_number -> list of [xtl, ytl, xbr, ybr] bounding boxes.
    A single frame may have multiple tracked objects (multiple tracks).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_boxes = {}  # {frame_idx: [[xtl, ytl, xbr, ybr], ...]}

    for track in root.findall("track"):
        for box in track.findall("box"):
            # Skip boxes where outside="1" (object not visible in this frame)
            if box.attrib.get("outside", "0") == "1":
                continue

            frame = int(box.attrib["frame"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            if frame not in frame_boxes:
                frame_boxes[frame] = []
            frame_boxes[frame].append([xtl, ytl, xbr, ybr])

    return frame_boxes


def discover_pairs(data_dir):
    """
    Scan data_dir for (.mov, .xml) pairs matched by base filename.
    Returns a list of (video_path, xml_path) tuples.
    Also checks for .mp4 files in case of mixed formats.
    """
    data_dir = Path(data_dir)
    video_extensions = {".mov", ".mp4", ".avi"}

    pairs = []
    for xml_file in data_dir.glob("*.xml"):
        stem = xml_file.stem
        for ext in video_extensions:
            video_file = data_dir / (stem + ext)
            if video_file.exists():
                pairs.append((str(video_file), str(xml_file)))
                break  # found a match, move on

    if not pairs:
        raise FileNotFoundError(
            f"No matched video+XML pairs found in '{data_dir}'. "
            "Make sure each .xml file has a matching .mov/.mp4 with the same base name."
        )

    return pairs


class BaseballVideoDataset(Dataset):
    """
    PyTorch Dataset for baseball bounding box detection.

    Each item returned is a tuple:
        image  : FloatTensor of shape (3, H, W), normalized to [0, 1]
        target : dict with keys:
                    'boxes'  : FloatTensor of shape (N, 4) in [xtl, ytl, xbr, ybr] format
                    'labels' : LongTensor of shape (N,) — all 1s (baseball class)
                    'image_id' : int

    Frames with no annotations are skipped.

    Args:
        data_dir  (str): Path to folder containing paired .mov and .xml files.
        transform (callable, optional): Additional transforms applied to the PIL image.
        img_size  (tuple): Resize all frames to (H, W). Default (720, 1280) matches source videos.
    """

    def __init__(self, data_dir, transform=None, img_size=(720, 1280)):
        self.img_size = img_size
        self.transform = transform

        # Discover all video+annotation pairs
        pairs = discover_pairs(data_dir)

        # Build flat list of (video_path, frame_idx, boxes)
        self.samples = []
        for video_path, xml_path in pairs:
            frame_boxes = parse_cvat_xml(xml_path)
            for frame_idx, boxes in frame_boxes.items():
                self.samples.append((video_path, frame_idx, boxes))

        if not self.samples:
            raise ValueError("Dataset is empty — no annotated frames found.")

        print(f"[Dataset] Found {len(pairs)} video(s), {len(self.samples)} annotated frames total.")

        # Base transform: resize and convert to tensor
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),  # scales to [0, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, boxes = self.samples[idx]

        # Extract the specific frame from the video
        frame = self._read_frame(video_path, frame_idx)

        # Apply transforms
        image = self.base_transform(frame)
        if self.transform:
            image = self.transform(image)

        # Scale bounding boxes if we resized the image
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.img_size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        scaled_boxes = []
        for box in boxes:
            xtl, ytl, xbr, ybr = box
            scaled_boxes.append([
                xtl * scale_x,
                ytl * scale_y,
                xbr * scale_x,
                ybr * scale_y,
            ])

        target = {
            "boxes": torch.tensor(scaled_boxes, dtype=torch.float32),
            "labels": torch.ones(len(scaled_boxes), dtype=torch.int64),  # class 1 = baseball
            "image_id": torch.tensor([idx]),
        }

        return image, target

    def _read_frame(self, video_path, frame_idx):
        """Open a video file and extract a single frame by index."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(
                f"Could not read frame {frame_idx} from '{video_path}'. "
                "Check that the video file is not corrupted."
            )

        # cv2 reads BGR — convert to RGB for torchvision
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def collate_fn(batch):
    """
    Custom collate function required for Faster R-CNN.
    Faster R-CNN expects a list of (image, target) tuples, not stacked tensors,
    because images may vary in size and each target has a variable number of boxes.
    """
    return tuple(zip(*batch))


def get_dataloader(data_dir, batch_size=2, shuffle=True, num_workers=0):
    """
    Convenience function to get a ready-to-use DataLoader.

    Args:
        data_dir   (str): Path to data folder with paired .mov + .xml files.
        batch_size (int): Number of frames per batch.
        shuffle    (bool): Shuffle samples each epoch.
        num_workers(int): Parallel workers. Use 0 on Windows or if debugging.

    Returns:
        DataLoader, Dataset
    """
    dataset = BaseballVideoDataset(data_dir=data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader, dataset


# ── Quick smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"Testing DataLoader on: {data_dir}")

    loader, dataset = get_dataloader(data_dir, batch_size=2)
    images, targets = next(iter(loader))

    print(f"Batch size      : {len(images)}")
    print(f"Image shape     : {images[0].shape}")
    print(f"Boxes in frame 0: {targets[0]['boxes']}")
    print(f"Labels          : {targets[0]['labels']}")
    print("DataLoader OK!")
