"""
Quick script to trim video to first 40 seconds
"""
import cv2
import sys

def trim_video(input_path, output_path, duration_seconds=40):
    """Trim video to specified duration"""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Calculate target frames
    target_frames = fps * duration_seconds
    target_frames = min(target_frames, total_frames)

    print(f"Trimming to {duration_seconds}s ({target_frames} frames)")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while frame_idx < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{target_frames} frames")

    cap.release()
    out.release()

    print(f"✅ Created {output_path} ({frame_idx} frames)")
    return True

if __name__ == "__main__":
    input_video = "Soccernet/SN-BAS-2024/challenge_england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City/224p.mp4"
    output_video = "challenge_clip_40s.mp4"

    trim_video(input_video, output_video, duration_seconds=40)
