import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any


SPLIT_MAP = {0: "train", 1: "eval", 2: "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert boundary predictions into per-video start/end segments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to scores.pkl, predictions.pkl, preds.pkl, or another pickle file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path. Supported suffixes: .csv, .jsonl",
    )
    parser.add_argument(
        "--info-pkl",
        type=Path,
        default=None,
        help="Optional info.pkl for attaching split/fps metadata.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied when the pickle contains boundary scores.",
    )
    parser.add_argument(
        "--num-in-frames",
        type=int,
        default=16,
        help="Window size used to produce each prediction.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="video",
        help="Fallback name when the pickle contains only a single unnamed sequence.",
    )
    parser.add_argument(
        "--segment-type",
        choices=["sign", "boundary"],
        default="sign",
        help="Export contiguous non-boundary sign segments or contiguous boundary runs.",
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


def to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def sequence_to_ints(sequence: Any, threshold: float) -> list[int]:
    values = [normalize_scalar(item) for item in to_list(sequence)]
    if not values:
        return []

    if any(isinstance(item, float) for item in values):
        return [1 if float(item) > threshold else 0 for item in values]
    return [int(item) for item in values]


def load_info(info_pkl: Path | None) -> dict[str, dict[str, Any]]:
    if info_pkl is None:
        return {}

    with info_pkl.open("rb") as handle:
        info_data = pickle.load(handle)

    videos = info_data["videos"]
    names = [normalize_scalar(item) for item in to_list(videos["name"])]
    splits = [normalize_scalar(item) for item in to_list(videos["split"])]
    fps_list = [normalize_scalar(item) for item in to_list(videos["videos"]["fps"])]
    org_names = [normalize_scalar(item) for item in to_list(videos.get("org_name", [""] * len(names)))]
    clip_starts = [normalize_scalar(item) for item in to_list(videos.get("start", [None] * len(names)))]
    clip_ends = [normalize_scalar(item) for item in to_list(videos.get("end", [None] * len(names)))]

    info_by_video = {}
    for idx, video_name in enumerate(names):
        split_id = splits[idx] if idx < len(splits) else None
        info_by_video[video_name] = {
            "split_id": split_id,
            "split": SPLIT_MAP.get(split_id, split_id),
            "fps": fps_list[idx] if idx < len(fps_list) else None,
            "org_name": org_names[idx] if idx < len(org_names) else "",
            "clip_start_sec": clip_starts[idx] if idx < len(clip_starts) else None,
            "clip_end_sec": clip_ends[idx] if idx < len(clip_ends) else None,
        }
    return info_by_video


def iter_sequences(payload: Any, threshold: float, fallback_name: str):
    payload = to_list(payload)

    if isinstance(payload, dict):
        first_value = next(iter(payload.values())) if payload else None
        if isinstance(first_value, dict) and ("scores" in first_value or "preds" in first_value):
            for video_name, item in payload.items():
                source_key = "preds" if "preds" in item else "scores"
                yield str(video_name), sequence_to_ints(item[source_key], threshold), source_key
            return

        for video_name, item in payload.items():
            yield str(video_name), sequence_to_ints(item, threshold), "sequence"
        return

    if isinstance(payload, list):
        yield fallback_name, sequence_to_ints(payload, threshold), "sequence"
        return

    raise TypeError(f"Unsupported pickle structure: {type(payload)}")


def contiguous_runs(sequence: list[int], target_value: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(sequence):
        if value == target_value and start is None:
            start = idx
        elif value != target_value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(sequence)))
    return runs


def make_rows(
    video_name: str,
    sequence: list[int],
    info_by_video: dict[str, dict[str, Any]],
    source_key: str,
    center_offset: int,
    segment_type: str,
) -> list[dict[str, Any]]:
    target_value = 0 if segment_type == "sign" else 1
    fps = info_by_video.get(video_name, {}).get("fps")
    rows = []

    for segment_index, (start_idx, end_idx) in enumerate(contiguous_runs(sequence, target_value)):
        start_frame = start_idx + center_offset
        end_frame_exclusive = end_idx + center_offset
        row = {
            "video_name": video_name,
            "segment_index": segment_index,
            "segment_type": segment_type,
            "source_key": source_key,
            "start_index_aligned": start_idx,
            "end_index_exclusive_aligned": end_idx,
            "end_index_inclusive_aligned": end_idx - 1,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame_exclusive,
            "end_frame_inclusive": end_frame_exclusive - 1,
            "length_aligned": end_idx - start_idx,
            "length_frames": end_idx - start_idx,
            "start_sec": (start_frame / fps) if fps else None,
            "end_sec": (end_frame_exclusive / fps) if fps else None,
        }
        row.update(info_by_video.get(video_name, {}))
        rows.append(row)

    return rows


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        fieldnames = [
            "video_name",
            "split",
            "split_id",
            "org_name",
            "clip_start_sec",
            "clip_end_sec",
            "fps",
            "segment_index",
            "segment_type",
            "source_key",
            "start_index_aligned",
            "end_index_exclusive_aligned",
            "end_index_inclusive_aligned",
            "start_frame",
            "end_frame_exclusive",
            "end_frame_inclusive",
            "length_aligned",
            "length_frames",
            "start_sec",
            "end_sec",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    raise ValueError("Output must end with .csv or .jsonl")


def main() -> None:
    args = parse_args()
    center_offset = args.num_in_frames // 2

    with args.input.open("rb") as handle:
        payload = pickle.load(handle)

    info_by_video = load_info(args.info_pkl)

    rows = []
    for video_name, sequence, source_key in iter_sequences(
        payload=payload,
        threshold=args.threshold,
        fallback_name=args.video_name,
    ):
        rows.extend(
            make_rows(
                video_name=video_name,
                sequence=sequence,
                info_by_video=info_by_video,
                source_key=source_key,
                center_offset=center_offset,
                segment_type=args.segment_type,
            )
        )

    write_rows(rows, args.output)
    print(f"Exported {len(rows)} {args.segment_type} segments to {args.output}")


if __name__ == "__main__":
    main()
