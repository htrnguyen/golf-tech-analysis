# Generated from: final_golf_swing.ipynb
# Converted at: 2026-01-01T14:06:29.903Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### 📦 Installation Summary
# 
# This code installs essential Python libraries for video processing, pose detection, progress tracking, and data handling:
# 
# - `moviepy==1.0.3`: Used for video editing (cutting, merging, adding audio).
# - `mediapipe==0.10.21`: Provides pose and hand detection (great for analyzing movement like golf swings).
# - `tqdm==4.67.1`: Adds progress bars to loops and downloads.
# - `dataclasses-json==0.6.7`: Allows easy serialization/deserialization of Python dataclasses to/from JSON.
# - `backoff==2.2.1`: Helps with retry logic for functions that might fail (e.g., API calls).
# - `supervision==0.26.0`: Useful for drawing overlays on video frames (bounding boxes, labels).
# - `yt-dlp`: Downloads videos from YouTube and other platforms (quietly installed with `-q`).
# 
# Note: `--no-deps` skips installing dependencies for the first command to avoid version conflicts.
# 


!pip install --no-deps moviepy==1.0.3 mediapipe==0.10.21 tqdm==4.67.1
!pip install dataclasses-json==0.6.7
!pip install backoff==2.2.1
!pip install supervision==0.26.0
!!pip install -q yt-dlp

# ### 🎬 YouTube Shorts Downloader (with `yt-dlp`)
# 
# This code uses `yt-dlp` to download high-quality YouTube Shorts as `.mp4` files:
# 
# - `-f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"`: Selects the best available video stream in MP4 format and merges it with the best audio in M4A. Falls back to a single MP4 if needed.
# - `-o "filename.mp4"`: Specifies the output filename for the downloaded video.
# - URLs point to YouTube Shorts featuring golfers Ludvig Åberg and Max Homa.
# 
# 📥 Output files:
# - `ludvig_aberg_driver.mp4`
# - `max_homa_iron.mp4`
# 


!yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "ludvig_aberg_driver.mp4" "https://www.youtube.com/shorts/9nhFobbdJxQ"
!yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "max_homa_iron.mp4" "https://www.youtube.com/shorts/aB__zg42i9g"


# ### 🧠 Core Imports Overview
# 
# This block imports essential libraries for processing, visualization, and media handling:
# 
# - **System & Utilities**:  
#   - `gc`, `time`, `subprocess`: For garbage collection, timing operations, and running shell commands.
# 
# - **Numerical & Signal Processing**:  
#   - `numpy`: Core library for numerical computations.  
#   - `uniform_filter1d` (from `scipy.ndimage`): Smooths 1D signals (e.g., joint positions over frames).
# 
# - **Visualization**:  
#   - `matplotlib.pyplot`: For plotting graphs and visuals.  
#   - `FigureCanvas`: Renders plots into images.  
#   - `mpl_use`: Sets a Colab-safe backend for rendering plots.
# 
# - **Media Processing**:  
#   - `cv2`: OpenCV library for reading, processing, and displaying video frames.  
#   - `mediapipe`: Used for pose estimation and landmark detection.
# 
# - **Colab Display**:  
#   - `IPython.display`: For showing videos inline using `Video()` and `display()` functions.
# 


# Core
import gc
import time
import subprocess

# Numerical and signal processing
import numpy as np
from scipy.ndimage import uniform_filter1d

# Visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import use as mpl_use  # Needed for Colab-safe rendering

# Media processing
import cv2
import mediapipe as mp

# Colab display tools
from IPython.display import Video, display


# ### 🔇 Add Silent Audio to Videos Using FFmpeg
# 
# This code prepares four golf swing videos (two input clips and two YouTube references) by adding a silent audio track to each. This ensures compatibility with video players that require an audio stream.
# 
# - **Input Videos**:
#   - `video_iron`, `video_driver`: Your own recorded swing clips.
#   - `iron_reference`, `driver_reference`: Reference swings from YouTube (e.g., Max Homa, Ludvig Åberg).
# 
# - **Output Paths**:
#   - For each video, a `_fixed.MP4` version is created with silent audio.
# 
# - **FFmpeg Command**:
#   - `-i <input>`: Input video file.
#   - `-f lavfi -i anullsrc`: Adds silent stereo audio at 44.1kHz.
#   - `-shortest`: Ends the audio when video ends.
#   - `-c:v libx264`: Encodes video using H.264.
#   - `-c:a aac -b:a 128k`: Encodes audio using AAC at 128 kbps.
#   - `-movflags +faststart`: Optimizes video for fast playback start.
# 
# 📁 Outputs:  
# - `*_fixed.MP4` files for all input and reference videos with embedded silent audio.
# 


# Input video path and define output path for audio-fixed video

iron_reference = "max_homa_iron.mp4"
driver_reference = "ludvig_aberg_driver"

video_iron = "/content/IMG_1022.MP4"  # original input video   # IRON
video_iron_fixed = video_iron.rsplit(".", 1)[0] + "_fixed.MP4"

video_driver = "/content/IMG_0758.MP4"  # alternate input # DRIVER
video_driver_fixed = video_driver.rsplit(".", 1)[0] + "_fixed.MP4"

iron_reference = "/content/max_homa_iron.mp4"
iron_reference_fixed = iron_reference.rsplit(".", 1)[0] + "_fixed.MP4"

driver_reference = "/content/ludvig_aberg_driver.mp4"
driver_reference_fixed = driver_reference.rsplit(".", 1)[0] + "_fixed.MP4"


# 1. Preserve original video and add silent audio track to a copy
ffmpeg_cmd = f"""
ffmpeg -y -i "{video_iron}" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
-shortest -c:v libx264 -c:a aac -b:a 128k -movflags +faststart "{video_iron_fixed}"
"""
subprocess.run(ffmpeg_cmd, shell=True)

# 1. Preserve original video and add silent audio track to a copy
ffmpeg_cmd = f"""
ffmpeg -y -i "{video_driver}" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
-shortest -c:v libx264 -c:a aac -b:a 128k -movflags +faststart "{video_driver_fixed}"
"""
subprocess.run(ffmpeg_cmd, shell=True)

# 1. Preserve original video and add silent audio track to a copy
ffmpeg_cmd = f"""
ffmpeg -y -i "{iron_reference}" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
-shortest -c:v libx264 -c:a aac -b:a 128k -movflags +faststart "{iron_reference_fixed}"
"""
subprocess.run(ffmpeg_cmd, shell=True)

# 1. Preserve original video and add silent audio track to a copy
ffmpeg_cmd = f"""
ffmpeg -y -i "{driver_reference}" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
-shortest -c:v libx264 -c:a aac -b:a 128k -movflags +faststart "{driver_reference_fixed}"
"""
subprocess.run(ffmpeg_cmd, shell=True)

# Preview the audio-fixed video (same content as original, with an added audio stream)
Video(video_iron_fixed, embed=True, width=600, height=800)

# Preview the audio-fixed video (same content as original, with an added audio stream)
Video(video_driver_fixed, embed=True, width=600, height=800)

# Preview the audio-fixed video (same content as original, with an added audio stream)
Video(iron_reference_fixed, embed=True, width=600, height=800)

# Preview the audio-fixed video (same content as original, with an added audio stream)
Video(driver_reference_fixed, embed=True, width=600, height=800)

# ### 🏌️‍♂️ Pose Feature Extraction (Wrist Y-Coordinate)
# 
# This function `extract_pose_features_debug(video_path)` analyzes a video frame-by-frame to extract the **vertical (Y) position of visible wrists** using MediaPipe Pose:
# 
# - **Video Setup**:
#   - Loads the video and retrieves its frame count and resolution.
#   - Initializes MediaPipe Pose in video (non-static) mode.
# 
# - **Frame Loop**:
#   - For each frame:
#     - Converts BGR to RGB for pose processing.
#     - Detects body landmarks using MediaPipe.
#     - If pose is found:
#       - Checks visibility of **left and right wrists**.
#       - Computes average Y-position (vertical) of visible wrists (scaled to video height).
#     - If no pose or wrist visibility is low, logs a warning and sets `wrist_y` as `None`.
# 
# - **Return**:
#   - Returns a list of dictionaries like `{"frame_idx": 42, "wrist_y": 138.5}` for each frame.
# 
# 👀 Use this to analyze hand motion over time, useful for swing segmentation or performance analysis.
# 


def extract_pose_features_debug(video_path):
    cap = cv2.VideoCapture(video_path)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total frames: {num_frames}, Resolution: {original_width}x{original_height}")

    used_height = original_height  # No rotation, use original height

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    frame_data = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i}")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        data = {"frame_idx": i}

        if results.pose_landmarks:
            # print(f"✅ Pose detected on frame {i}")
            lm = results.pose_landmarks.landmark
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            visible_wrists = [p for p in (lw, rw) if p.visibility > 0.4]
            if visible_wrists:
                avg_y = np.mean([p.y for p in visible_wrists])
                data["wrist_y"] = avg_y * used_height
                # print(f"👋 Frame {i}: Wrist Y = {data['wrist_y']:.2f}")
            else:
                print(f"⚠️ Frame {i}: Wrists not visible")
                data["wrist_y"] = None
        else:
            print(f"❌ Frame {i}: No pose detected")
            data["wrist_y"] = None

        frame_data.append(data)

    cap.release()
    pose.close()
    return frame_data




# ### 📈 Run Pose Extraction on All Videos
# 
# This block runs the `extract_pose_features_debug()` function on all four prepared videos (with silent audio):
# 
# - `video_iron_fixed`: Your iron swing clip  
# - `video_driver_fixed`: Your driver swing clip  
# - `iron_reference_fixed`: Max Homa’s reference swing  
# - `driver_reference_fixed`: Ludvig Åberg’s reference swing  
# 
# Each call extracts the **Y-coordinate of visible wrists** frame-by-frame and stores the results in:
# 
# - `frame_data_iron`
# - `frame_data_driver`
# - `frame_data_iron_reference`
# - `frame_data_driver_reference`
# 
# 📦 These variables now hold pose-based wrist motion data for later analysis or swing comparison.
# 


frame_data_iron = extract_pose_features_debug(video_iron_fixed)
frame_data_driver = extract_pose_features_debug(video_driver_fixed)
frame_data_iron_reference = extract_pose_features_debug(iron_reference_fixed)
frame_data_driver_reference = extract_pose_features_debug(driver_reference_fixed)

# 


# ### 🏌️‍♀️ Golf Swing Phase Detection via Wrist Y-Motion
# 
# This function `detect_swing_phases()` takes wrist Y-coordinate data and dynamically identifies key golf swing phases using smoothed motion patterns and gradient analysis:
# 
# #### 🔍 Key Steps:
# - **Smoothing & Velocity**:
#   - Applies a moving average to wrist Y (`smoothing_window`), then calculates velocity and its magnitude.
# 
# - **Swing Start & End**:
#   - Detects when motion exceeds a dynamic threshold (`threshold_percentile`) to mark swing boundaries.
# 
# - **Address Detection**:
#   - Scans backwards from swing start to find a stable ("flat") wrist phase, indicating the pre-swing address.
# 
# - **Top of Backswing**:
#   - Finds the lowest wrist Y point before the swing starts descending (indicating the top of the backswing).
# 
# - **Impact Detection**:
#   - After the top, identifies a stable high point in wrist Y as impact, using tolerance around local max.
# 
# - **Phase Segmentation**:
#   - Returns six labeled segments: `"Address"`, `"Backswing"`, `"Top"`, `"Downswing"`, `"Impact"`, `"Follow Through"`.
# 
# #### 📊 Visualization:
# - Plots wrist trajectory with Y flipped (so upward motion looks natural).
# - Highlights each swing phase using colored spans and labels.
# 
# #### 📤 Returns:
# - `phase_ranges`: Dictionary of start/end frames for each phase.
# - `swing_start`, `swing_end`: Frame indices marking full swing.
# - `wrist_y`, `smoothed`: Raw and smoothed wrist Y arrays for further analysis.
# 
# ✅ This unified logic handles both pros and amateurs, dynamically adjusting to wrist movement stability and swing patterns.
# 




def detect_swing_phases(frame_data, smoothing_window=5, precheck_window=30, threshold_percentile=90):
    """
    Analyze wrist Y trajectory and determine key golf swing phase ranges.
    Unified logic. Dynamically determines Address using flatness in wrist_y.
    """
    # Convert to numpy and handle None
    wrist_y = np.array([d["wrist_y"] if d["wrist_y"] is not None else np.nan for d in frame_data], dtype=np.float32)
    smoothed = uniform_filter1d(wrist_y, size=smoothing_window, mode='nearest')
    velocity = np.gradient(smoothed)
    velocity_mag = np.abs(velocity)

    # Motion threshold and swing start
    early_motion = np.nanmean(velocity_mag[:precheck_window])
    motion_std = np.nanstd(velocity_mag[:precheck_window])
    buffer_start = precheck_window if (early_motion < 0.001 and motion_std < 0.001) else 0
    threshold = np.nanpercentile(velocity_mag[buffer_start:], threshold_percentile)
    motion_indices = np.where(velocity_mag > threshold)[0]
    motion_indices = motion_indices[motion_indices > buffer_start]

    if len(motion_indices) > 0:
        swing_start = int(motion_indices[0])
        peak_idx = motion_indices[np.argmax(velocity_mag[motion_indices])]
        target_y = smoothed[swing_start]
        post_peak_range = smoothed[peak_idx+1:]
        swing_end = peak_idx + 1 + int(np.nanargmin(np.abs(post_peak_range - target_y))) if len(post_peak_range) > 0 else len(wrist_y) - 1
    else:
        swing_start = 0
        swing_end = len(wrist_y) - 1

    # === Dynamic Address detection ===
    flat_std_thresh = 1.0
    min_flat_frames = 10
    address_start = 0
    for i in range(swing_start - min_flat_frames, 0, -1):
        window = smoothed[i:swing_start]
        if np.count_nonzero(~np.isnan(window)) < min_flat_frames:
            continue
        if np.nanstd(window) < flat_std_thresh:
            address_start = i
        else:
            break
    address_end = swing_start

    # === Top of backswing ===
    full_swing_segment = smoothed[swing_start:swing_end]
    diff = np.diff(full_swing_segment)
    rising_indices = np.where(diff > 0)[0]
    top_candidate_max = swing_start + rising_indices[0] if len(rising_indices) > 0 else min(swing_start + 20, len(smoothed) - 1)
    top_search_end = min(top_candidate_max, swing_end)
    top_idx = swing_start + int(np.nanargmin(smoothed[swing_start:top_search_end])) if top_search_end > swing_start else swing_start
    top_range = (max(0, top_idx - 5), min(len(wrist_y) - 1, top_idx + 5))

    # === Impact ===
    post_top = smoothed[top_range[1]:swing_end]
    if len(post_top) == 0 or np.all(np.isnan(post_top)):
        impact_range = (top_range[1], top_range[1] + 1)
    else:
        local_max_idx = int(np.nanargmax(post_top))
        max_val = post_top[local_max_idx]
        local_min = np.nanmin(post_top)
        local_max = np.nanmax(post_top)
        dynamic_tol = 0.05 * (local_max - local_min)
        left = local_max_idx
        right = local_max_idx
        while left > 0 and abs(post_top[left - 1] - max_val) < dynamic_tol:
            left -= 1
        while right < len(post_top) - 1 and abs(post_top[right + 1] - max_val) < dynamic_tol:
            right += 1
        impact_range = (top_range[1] + left, top_range[1] + right)

    # === Phase dictionary ===
    phase_ranges = {
        "Address": (address_start, address_end),
        "Backswing": (swing_start, top_range[0]),
        "Top": top_range,
        "Downswing": (top_range[1], impact_range[0]),
        "Impact": impact_range,
        "Follow Through": (impact_range[1], swing_end)
    }

    # === Plotting with gaps and flipped Y ===
    x = np.array([d['frame_idx'] for d in frame_data])
    y = np.array([d['wrist_y'] if d['wrist_y'] is not None else np.nan for d in frame_data], dtype=np.float32)
    valid_y = y[~np.isnan(y)]
    y_flipped = np.nan if len(valid_y) == 0 else np.max(valid_y) - y + np.min(valid_y)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y_flipped, label='Wrist Y (flipped)', color='black')
    colors = ['gray', 'blue', 'orange', 'green', 'red', 'purple']
    for i, (phase, (start, end)) in enumerate(phase_ranges.items()):
        plt.axvspan(start, end, alpha=0.3, color=colors[i % len(colors)], label=phase)
    plt.xlabel("Frame Index")
    plt.ylabel("Wrist Y (flipped)")
    plt.title("Wrist Trajectory with Dynamic Phase Detection")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return phase_ranges, swing_start, swing_end, wrist_y, smoothed


phase_ranges_iron, swing_start_iron, swing_end_iron, wrist_y_iron, smoothed_iron = detect_swing_phases(frame_data_iron)
print("🏌️‍♂️ Detected Phase Ranges:", phase_ranges_iron)

phase_ranges_driver, swing_start_driver, swing_end_driver, wrist_y_driver, smoothed_driver = detect_swing_phases(frame_data_driver)
print("🏌️‍♂️ Detected Phase Ranges:", phase_ranges_driver)

phase_ranges_iron_reference, swing_start, swing_end, wrist_y, smoothed = detect_swing_phases(frame_data_iron_reference)
print("🏌️‍♂️ Detected Phase Ranges:", phase_ranges_iron_reference)

phase_ranges_driver_reference, swing_start, swing_end, wrist_y, smoothed = detect_swing_phases(frame_data_driver_reference)
print("🏌️‍♂️ Detected Phase Ranges:", phase_ranges_driver_reference)

# ### 🧑‍🎓📹 Pro vs Student Swing Comparison by Phase (with Pose Overlay)
# 
# The `compare_swing_phases()` function visually compares each swing phase between a **student video** and a **pro reference video**, displaying side-by-side frames (optionally with pose landmarks).
# 
# #### 📦 Inputs:
# - `student_video_path`, `pro_video_path`: Paths to fixed videos (with silent audio).
# - `student_phases`, `pro_phases`: Dicts of frame ranges per phase (from `detect_swing_phases()`).
# - `show_pose`: If `True`, overlays pose landmarks using MediaPipe.
# 
# #### 🧠 Core Logic:
# - For each swing phase (e.g., `"Address"`, `"Impact"`):
#   - Finds the middle frame of that phase for both student and pro.
#   - Extracts and optionally annotates frames using MediaPipe Pose.
#   - Converts BGR → RGB for display compatibility.
# 
# #### 🎨 Output:
# - A multi-row Matplotlib figure with:
#   - Left: Student frame for each phase.
#   - Right: Pro frame for the same phase.
#   - Optional pose overlay to help analyze alignment, posture, and joint positioning.
# 
# #### 🧹 Cleanup:
# - Releases video capture resources and closes the MediaPipe model.
# 
# ✅ Use this for **visual feedback and coaching**, making swing differences easy to spot phase-by-phase.
# 




def compare_swing_phases(
    student_video_path, student_phases,
    pro_video_path, pro_phases,
    show_pose=True
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    def get_frame(cap, frame_idx, apply_pose=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"❌ Failed to read frame {frame_idx}")
        if apply_pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                )
        return frame

    student_cap = cv2.VideoCapture(student_video_path)
    pro_cap = cv2.VideoCapture(pro_video_path)

    phase_names = list(student_phases.keys())
    num_phases = len(phase_names)

    plt.figure(figsize=(16, 6 * num_phases))

    for i, phase in enumerate(phase_names):
        # === Frame indices ===
        s_start, s_end = student_phases[phase]
        p_start, p_end = pro_phases[phase]

        s_middle = int((s_start + s_end) // 2)
        p_middle = int((p_start + p_end) // 2)

        # === Load frames ===
        student_frame = get_frame(student_cap, s_middle, apply_pose=show_pose)
        pro_frame = get_frame(pro_cap, p_middle, apply_pose=show_pose)

        student_rgb = cv2.cvtColor(student_frame, cv2.COLOR_BGR2RGB)
        pro_rgb = cv2.cvtColor(pro_frame, cv2.COLOR_BGR2RGB)

        # === Plot ===
        plt.subplot(num_phases, 2, 2*i+1)
        plt.imshow(student_rgb)
        plt.title(f"Student - {phase}")
        plt.axis("off")

        plt.subplot(num_phases, 2, 2*i+2)
        plt.imshow(pro_rgb)
        plt.title(f"Pro - {phase}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    student_cap.release()
    pro_cap.release()
    pose.close()


compare_swing_phases(
    student_video_path=video_iron_fixed,
    student_phases=phase_ranges_iron,
    pro_video_path=iron_reference_fixed,
    pro_phases=phase_ranges_iron_reference,
    show_pose=True  # Set to False to skip pose overlay
)


compare_swing_phases(
    student_video_path=video_driver_fixed,
    student_phases=phase_ranges_driver,
    pro_video_path=driver_reference_fixed,
    pro_phases=phase_ranges_driver_reference,
    show_pose=True  # Set to False to skip pose overlay
)


# ### 🎥 Generate Annotated Debug Video with Swing Overlay and Slow Motion
# 
# This function `generate_debug_video_with_overlay()` creates a side-by-side **video+plot debug visualization**, showing real-time wrist motion trajectory alongside the actual video frames.
# 
# ---
# 
# #### 🔁 Step-by-Step Workflow:
# 
# - **Frame-by-Frame Rendering**:
#   - Reads each frame from the input video.
#   - Overlays a Matplotlib plot of smoothed wrist Y-position (`smoothed`), marks current frame, swing start/end, and displays swing phase label (via `get_swing_phase_label()`).
# 
# - **Combined Output**:
#   - Horizontally stacks video frame + debug plot.
#   - Writes to an MP4 video using OpenCV (`cv2.VideoWriter`).
# 
# - **Plot Details**:
#   - Flipped Y-axis for intuitive visualization.
#   - Clear phase zones (`Address`, `Backswing`, etc.) shown with labels and color spans.
# 
# ---
# 
# #### 🐢 Slow-Mo Re-encoding:
# - After video is saved, re-encodes it using **FFmpeg** to:
#   - Insert a slow-motion segment during the swing window (`swing_start` → `swing_end`).
#   - Concatenate trimmed clips before, during (slowed), and after the swing.
#   - Save to a more compatible format (`*_playable.mp4`).
# 
# ---
# 
# #### 📤 Returns:
# - A Colab-embedded video (`IPython.display.Video`) showing:
#   - Original clip + live tracking plot.
#   - Frame-wise swing phase.
#   - Smooth debug overlay and swing window.
# 
# 🎯 Perfect for **visual diagnostics**, **student feedback**, and **explaining biomechanics** with frame-accurate debug context.
# 



# Get swing phase label for a given frame index
def get_swing_phase_label(frame_idx, phase_ranges):
    for label, (start, end) in phase_ranges.items():
        if start <= frame_idx <= end:
            return label
    return None

def generate_debug_video_with_overlay(
    video_filename: str,
    output_path: str,
    wrist_y: list,
    smoothed: list,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int
):
    import gc
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib import use as mpl_use
    mpl_use("Agg")  # ✅ For headless backend in Colab

    cap = cv2.VideoCapture(video_filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Frame resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")

    plot_width = int(width * 2.0)
    output_size = (width + plot_width, height)

    # ✅ Use mp4v codec for writing temporary mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    fig, ax = plt.subplots(figsize=(plot_width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvas(fig)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    print("🔄 Rendering video with overlaid debug plots...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx >= len(wrist_y):
            print(f"📌 Reached end of trajectory data at frame {frame_idx}.")
            break

        try:
            ax.clear()
            ax.plot(smoothed, label="Smoothed Wrist Y", linestyle='--', color='blue')
            ax.axvline(frame_idx, color='black', linestyle='-', label="Current Frame")

            # ✅ Phase label helper (you must define or import this)
            phase_label = get_swing_phase_label(frame_idx, phase_ranges)
            if phase_label:
                ax.text(0.02, 0.95, phase_label, transform=ax.transAxes,
                        fontsize=18, color='black', fontweight='bold',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            ax.axvspan(swing_start, swing_end, color='green', alpha=0.2, label="Swing Window")

            ax.set_title(f"Debug Plot (Frame {frame_idx})", fontsize=20)
            ax.set_xlabel("Frame", fontsize=16)
            ax.set_ylabel("Wrist Y Position", fontsize=16)
            ax.set_xlim(0, len(wrist_y))
            ax.set_ylim(np.nanmax(wrist_y) + 10, np.nanmin(wrist_y) - 10)
            ax.legend(fontsize=14)
            fig.tight_layout()
            canvas.draw()

            plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
                canvas.get_width_height()[1], canvas.get_width_height()[0], 4
            )[..., :3]

            plot_img = cv2.resize(plot_img, (plot_width, height))
            frame = cv2.resize(frame, (width, height))
            combined_frame = np.hstack((frame, plot_img))
            out.write(combined_frame)

            if frame_idx % 50 == 0:
                print(f"✅ Processed frame {frame_idx}/{total_frames}")

            frame_idx += 1
            gc.collect()

        except Exception as e:
            print(f"❌ Error at frame {frame_idx}: {e}")
            break

    cap.release()
    out.release()
    print(f"✅ Debug video saved as {output_path}")

    # === Re-encode with FFmpeg ===
    reencoded_path = output_path.replace(".mp4", "_playable.mp4")

    # Convert swing frame indices to seconds
    start_time = swing_start / fps
    end_time = swing_end / fps
    slow_factor = 2.0  # Adjust this factor as needed

    # FFmpeg slow-motion filter logic
    filter_complex = (
        f"[0:v]trim=0:{start_time:.3f},setpts=PTS-STARTPTS[v1];"
        f"[0:v]trim={start_time:.3f}:{end_time:.3f},setpts={slow_factor}*(PTS-STARTPTS)[v2];"
        f"[0:v]trim={end_time:.3f},setpts=PTS-STARTPTS[v3];"
        f"[v1][v2][v3]concat=n=3:v=1:a=0[outv]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        reencoded_path
    ]

    print("🎬 Re-encoding with slow-motion for Colab compatibility...")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    # Optional: give disk a moment to flush
    time.sleep(1)

    return Video(reencoded_path, embed=True, width=960)


debug_video = generate_debug_video_with_overlay(video_iron_fixed, "iron_debug.mp4", wrist_y_iron, smoothed_iron, phase_ranges_iron, swing_start_iron, swing_end_iron)

debug_video


debug_video = generate_debug_video_with_overlay(video_driver_fixed, "iron_debug.mp4", wrist_y_driver, smoothed_driver, phase_ranges_driver, swing_start_driver, swing_end_driver)

debug_video