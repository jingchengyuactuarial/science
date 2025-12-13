import cv2
import numpy as np
import os

def merge_videos():
    v1_path = 'video1.mp4'
    v2_path = 'video2.mp4'
    out_path = 'combined_hero.mp4'

    if not os.path.exists(v1_path) or not os.path.exists(v2_path):
        print("Error: Input files not found.")
        return

    cap1 = cv2.VideoCapture(v1_path)
    cap2 = cv2.VideoCapture(v2_path)

    # Get properties from v1
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Check fps
    if fps == 0 or fps != fps: # NaN check just in case
        fps = 30.0
        
    print(f"Video 1 config: {w}x{h} @ {fps}fps")

    # Codec
    # Try avc1 (H.264) which is browser friendly. mp4v is often black in Chrome/Edge.
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        print("Warning: avc1 not found, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    trim_sec = 2.0
    fade_sec = 1.0 # 1 second crossfade for smoothness
    
    trim_frames = int(trim_sec * fps)
    fade_frames = int(fade_sec * fps)

    print(f"Trimming {trim_frames} frames from V2. Fading for {fade_frames} frames.")

    # 1. Read V1, keeping last 'fade_frames' in buffer
    frames_v1_buffer = []
    
    print("Reading Video 1...")
    frame_count1 = 0
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        
        # Resize if needed (robustness)
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            
        frames_v1_buffer.append(frame)
        
        # If buffer is full, write the popped item
        if len(frames_v1_buffer) > fade_frames:
            out.write(frames_v1_buffer.pop(0))
        frame_count1 += 1

    print(f"Video 1 read. Buffer size: {len(frames_v1_buffer)}")

    # 2. Skip frames in V2
    print("Skipping V2 intro...")
    for _ in range(trim_frames):
        ret = cap2.grab() # faster than read()
        if not ret:
            print("Video 2 is too short to trim!")
            break

    # 3. Blend transition
    print("Blending transition...")
    blended_count = 0
    for i in range(len(frames_v1_buffer)): # Use whatever is in buffer (should be fade_frames)
        ret, frame2 = cap2.read()
        if not ret:
            break

        if frame2.shape[1] != w or frame2.shape[0] != h:
            frame2 = cv2.resize(frame2, (w, h))
            
        frame1 = frames_v1_buffer[i]
        
        alpha = (i + 1) / (len(frames_v1_buffer) + 1) # simple linear
        # We want to fade OUT v1 (alpha 1 -> 0) means v1 * (1-alpha)??
        # Usually: Output = V1 * (1-t) + V2 * t (fade from V1 to V2)
        # alpha here goes 0 -> 1 approx
        
        beta = alpha 
        # t=0: Mostly V1. t=1: Mostly V2.
        
        blended = cv2.addWeighted(frame1, 1.0 - beta, frame2, beta, 0)
        out.write(blended)
        blended_count += 1
        
    print(f"Blended {blended_count} frames.")

    # 4. Write rest of V2
    print("Writing rest of Video 2...")
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        out.write(frame)

    cap1.release()
    cap2.release()
    out.release()
    print("Finished.")

if __name__ == "__main__":
    merge_videos()
