"""Frame extraction helpers (placeholder)."""
def extract_frames(video_path, out_dir, fps=1):
    """Placeholder: extract frames from `video_path` into `out_dir` at `fps`."""
    print(f"extract_frames: {video_path} -> {out_dir} @ {fps}fps")


if __name__ == "__main__":
    import sys
    extract_frames(sys.argv[1] if len(sys.argv)>1 else "video.mp4", "./frames")
