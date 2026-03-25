import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


SPLIT_MAP = {0: "train", 1: "eval", 2: "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export contiguous gloss segments from an info.pkl file."
    )
    parser.add_argument(
        "--info-pkl",
        type=Path,
        required=True,
        help="Path to data/info/<dataset>/info.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path. Use .csv or .jsonl.",
    )
    parser.add_argument(
        "--label-key",
        choices=["gloss", "gloss_id"],
        default="gloss",
        help="Frame-level alignment key used to define contiguous segments.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Label to skip. Can be passed multiple times, e.g. --exclude-label -1",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override. Defaults to the parent folder name of info.pkl.",
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def normalize_sequence(values: Iterable[Any]) -> list[Any]:
    return [normalize_scalar(item) for item in as_list(values)]


def uniform_value(values: list[Any], start: int, end: int) -> tuple[Any, bool]:
    chunk = [normalize_scalar(item) for item in values[start:end]]
    if not chunk:
        return None, True
    first = chunk[0]
    return first, all(item == first for item in chunk)


def segment_rows(info_data: dict[str, Any], dataset_name: str, label_key: str, excluded: set[str]):
    videos = info_data["videos"]
    alignments = videos["alignments"]

    names = normalize_sequence(videos["name"])
    splits = normalize_sequence(videos["split"])
    fps_list = normalize_sequence(videos["videos"]["fps"])
    org_names = normalize_sequence(videos.get("org_name", [""] * len(names)))
    clip_starts = normalize_sequence(videos.get("start", [None] * len(names)))
    clip_ends = normalize_sequence(videos.get("end", [None] * len(names)))

    primary = [normalize_sequence(item) for item in alignments[label_key]]
    gloss_seq = [normalize_sequence(item) for item in alignments.get("gloss", primary)]
    gloss_id_seq = [normalize_sequence(item) for item in alignments.get("gloss_id", primary)]

    for video_idx, video_name in enumerate(names):
        labels = primary[video_idx]
        if not labels:
            continue

        fps = fps_list[video_idx] if video_idx < len(fps_list) else None
        split_id = splits[video_idx]
        split_name = SPLIT_MAP.get(split_id, str(split_id))

        start = 0
        current = labels[0]
        segment_index = 0

        for frame_idx in range(1, len(labels) + 1):
            is_boundary = frame_idx == len(labels) or labels[frame_idx] != current
            if not is_boundary:
                continue

            label_text = str(normalize_scalar(current))
            if label_text not in excluded:
                gloss_value, gloss_uniform = uniform_value(gloss_seq[video_idx], start, frame_idx)
                gloss_id_value, gloss_id_uniform = uniform_value(
                    gloss_id_seq[video_idx], start, frame_idx
                )
                start_sec = start / fps if fps else None
                end_sec = frame_idx / fps if fps else None
                yield {
                    "dataset": dataset_name,
                    "split": split_name,
                    "split_id": split_id,
                    "video_name": video_name,
                    "org_name": org_names[video_idx] if video_idx < len(org_names) else "",
                    "clip_start_sec": clip_starts[video_idx] if video_idx < len(clip_starts) else None,
                    "clip_end_sec": clip_ends[video_idx] if video_idx < len(clip_ends) else None,
                    "fps": fps,
                    "segment_index": segment_index,
                    "label_key": label_key,
                    "segment_label": normalize_scalar(current),
                    "gloss": gloss_value,
                    "gloss_uniform": gloss_uniform,
                    "gloss_id": gloss_id_value,
                    "gloss_id_uniform": gloss_id_uniform,
                    "start_frame": start,
                    "end_frame_exclusive": frame_idx,
                    "end_frame_inclusive": frame_idx - 1,
                    "length_frames": frame_idx - start,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
                segment_index += 1

            if frame_idx < len(labels):
                start = frame_idx
                current = labels[frame_idx]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "split",
        "split_id",
        "video_name",
        "org_name",
        "clip_start_sec",
        "clip_end_sec",
        "fps",
        "segment_index",
        "label_key",
        "segment_label",
        "gloss",
        "gloss_uniform",
        "gloss_id",
        "gloss_id_uniform",
        "start_frame",
        "end_frame_exclusive",
        "end_frame_inclusive",
        "length_frames",
        "start_sec",
        "end_sec",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or args.info_pkl.parent.name

    with args.info_pkl.open("rb") as handle:
        info_data = pickle.load(handle)

    rows = list(
        segment_rows(
            info_data=info_data,
            dataset_name=dataset_name,
            label_key=args.label_key,
            excluded={str(item) for item in args.exclude_label},
        )
    )

    suffix = args.output.suffix.lower()
    if suffix == ".csv":
        write_csv(rows, args.output)
    elif suffix == ".jsonl":
        write_jsonl(rows, args.output)
    else:
        raise ValueError("Output must end with .csv or .jsonl")

    print(f"Exported {len(rows)} segments to {args.output}")


if __name__ == "__main__":
    main()
