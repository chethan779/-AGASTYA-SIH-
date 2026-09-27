import cv2
import time
import json
import threading
import queue
import pyttsx3

from datetime import datetime
from collections import deque

import torch
from ultralytics import YOLO

# ============================================================
# TEXT TO SPEECH (TTS) SYSTEM
# ============================================================

tts_queue = queue.Queue()


def tts_worker():
    """Background thread to process speech without dropping camera FPS."""

    engine = pyttsx3.init()
    engine.setProperty('rate', 200)

    while True:
        text = tts_queue.get()

        if text is None:
            break

        try:
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print("TTS Error:", e)

        finally:
            tts_queue.task_done()


tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()


def speak(text):
    """Adds text to the speech queue if it's not too backed up."""

    if tts_queue.qsize() < 3:
        tts_queue.put(text)


# ============================================================
# SEQUENCE LOGIC (NEW SECTION)
# ============================================================

# The required order of operations
REQUIRED_SEQUENCE = [
    ("yellow", "yellow"),
    ("orange", "orange"),
    ("pink", "pink")
]

current_step_index = 0

def get_next_step_text():
    """Returns the text for the current required step."""
    if current_step_index < len(REQUIRED_SEQUENCE):
        ball, box = REQUIRED_SEQUENCE[current_step_index]
        return f"The next step is {ball} ball in {box} box."
    else:
        return "All steps completed. Sequence finished."

def check_placement_sequence(actual_ball, actual_box):
    """
    Checks if the placed ball/box matches the required sequence.
    Returns: "CORRECT", "SKIPPED", or "WRONG"
    """
    global current_step_index

    # If already finished all steps
    if current_step_index >= len(REQUIRED_SEQUENCE):
        return "WRONG" # Or another status like "FINISHED"

    expected_ball, expected_box = REQUIRED_SEQUENCE[current_step_index]

    # 1. Exact match for current step
    if actual_ball == expected_ball and actual_box == expected_box:
        current_step_index += 1
        return "CORRECT"

    # 2. Check if they skipped ahead to a future step
    for i in range(current_step_index + 1, len(REQUIRED_SEQUENCE)):
        future_ball, future_box = REQUIRED_SEQUENCE[i]
        if actual_ball == future_ball and actual_box == future_box:
            return "SKIPPED"

    # 3. Otherwise, it's just wrong colors/boxes
    return "WRONG"


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "bigbang.pt"

CAMERA_ID = 0
CONFIDENCE = 0.30
IMAGE_SIZE = 416

# Your camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


# ------------------------------------------------------------
# Placement settings
# ------------------------------------------------------------

PLACEMENT_MARGIN = 180
CAPTURE_MARGIN = 80
BALL_OUTSIDE_FRAMES = 4
APPROACH_FRAMES_REQUIRED = 2
CONFIRMATION_TIME = 2.0
BOX_IGNORE_MARGIN = 50
MOVEMENT_TOLERANCE = 10
TRAJECTORY_LENGTH = 15


# ============================================================
# CLASSES
# ============================================================

BALL_CLASSES = {
    "yellow ball",
    "pink ball",
    "orange ball"
}

BOX_CLASSES = {
    "yellow box",
    "pink box",
    "orange box"
}


# ============================================================
# GPU / CPU
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    print("CUDA available: True")
    print("Using NVIDIA GPU")
else:
    DEVICE = "cpu"
    print("CUDA available: False")
    print("Using CPU")


# ============================================================
# STARTUP
# ============================================================

print()
print("==========================================")
print("             TARS STARTING")
print("==========================================")
print()
print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Classes:", model.names)
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center(box):
    x1, y1, x2, y2 = box
    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )

def inside(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )

def expand_box(box, margin):
    x1, y1, x2, y2 = box
    return (
        x1 - margin,
        y1 - margin,
        x2 + margin,
        y2 + margin
    )

def overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    return not (
        ax2 < bx1
        or
        ax1 > bx2
        or
        ay2 < by1
        or
        ay1 > by2
    )

def distance_to_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5

def color_of(class_name):
    if class_name.endswith(" ball"):
        return class_name[:-5]
    if class_name.endswith(" box"):
        return class_name[:-4]
    return class_name

def save_event(ball_class, box_class, result):
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assistant": "TARS",
        "ball": ball_class,
        "box": box_class,
        "result": result
    }

    try:
        try:
            with open("experiment_log.json", "r") as file:
                events = json.load(file)
        except Exception:
            events = []

        events.append(event)

        with open("experiment_log.json", "w") as file:
            json.dump(events, file, indent=4)
    except Exception as e:
        print("Could not save event:", e)


# ============================================================
# EXPERIMENT RESET
# ============================================================

def reset_experiment(reset_sequence_too=False):
    global locked, locked_box, locked_box_class, placement_box, capture_box
    global ball_started_outside, ball_outside_count, ball_crossed_placement, ball_color
    global trajectory, previous_distance, approach_count
    global ball_reached_capture, confirmation_started, confirmation_start
    global placement_result
    global current_step_index

    if reset_sequence_too:
        current_step_index = 0

    locked = False
    locked_box = None
    locked_box_class = None
    placement_box = None
    capture_box = None

    ball_started_outside = False
    ball_outside_count = 0
    ball_crossed_placement = False
    ball_color = None

    trajectory.clear()
    previous_distance = None
    approach_count = 0

    ball_reached_capture = False
    confirmation_started = False
    confirmation_start = None
    placement_result = None


# ============================================================
# EXPERIMENT VARIABLES
# ============================================================

locked = False
locked_box = None
locked_box_class = None
placement_box = None
capture_box = None

ball_started_outside = False
ball_outside_count = 0
ball_crossed_placement = False
ball_color = None

trajectory = deque(maxlen=TRAJECTORY_LENGTH)
previous_distance = None
approach_count = 0

ball_reached_capture = False
confirmation_started = False
confirmation_start = None

placement_result = None


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = cap.get(cv2.CAP_PROP_FPS)

print()
print("Camera:")
print("Width:", width)
print("Height:", height)
print("FPS:", camera_fps)
print()


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow("TARS - AI Experiment", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TARS - AI Experiment", 1600, 900)


# ============================================================
# START
# ============================================================

print("==========================================")
print("              TARS READY")
print("==========================================")
print()
print("Show a box.")
print("Press L to LOCK.")
print("Move a ball from OUTSIDE.")
print("Press N for NEXT STEP info.")
print("Press C to CLEAR.")
print("Press Q to QUIT.")
print()

message = "TARS: DETECT BOX"
status = "BOX IS NOT LOCKED"
speak("TARS is ready. Please follow the sequence.")

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        continue

    # Mirror camera
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # ========================================================
    # YOLO
    # ========================================================
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for detection in result.boxes:
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

    # ========================================================
    # SEPARATE BALLS AND BOXES
    # ========================================================
    visible_boxes = [d for d in detections if d["class"] in BOX_CLASSES]
    visible_balls = [d for d in detections if d["class"] in BALL_CLASSES]

    # ========================================================
    # KEYBOARD INPUT
    # ========================================================
    key = cv2.waitKeyEx(1)

    # --------------------------------------------------------
    # N = NEXT STEP (TTS Trigger)
    # --------------------------------------------------------
    if key in (ord("n"), ord("N")):
        next_step_text = get_next_step_text()
        print(f"TARS: {next_step_text}")
        speak(next_step_text)

    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------
    elif key in (ord("q"), ord("Q")):
        speak("Shutting down.")
        break

    # --------------------------------------------------------
    # C = CLEAR
    # --------------------------------------------------------
    elif key in (ord("c"), ord("C")):
        # Clearing completely resets the current lock, but we retain
        # the current sequence step unless they restart entirely.
        # (Pass True if you want 'c' to restart the whole 3-step sequence)
        reset_experiment(reset_sequence_too=False)

        message = "TARS: EXPERIMENT CLEARED"
        status = "SHOW A BOX AND PRESS L"

        print("\nTARS: Current lock cleared.\n")
        speak("Current placement cleared.")

    # --------------------------------------------------------
    # L = LOCK
    # --------------------------------------------------------
    elif key in (ord("l"), ord("L")):
        if not locked:
            if visible_boxes:
                selected = max(visible_boxes, key=lambda d: d["confidence"])
                locked_box = selected["box"]
                locked_box_class = selected["class"]

                placement_box = expand_box(locked_box, PLACEMENT_MARGIN)
                capture_box = expand_box(locked_box, CAPTURE_MARGIN)

                locked = True
                ball_started_outside = False
                ball_outside_count = 0
                ball_crossed_placement = False
                ball_color = None

                trajectory.clear()
                previous_distance = None
                approach_count = 0

                ball_reached_capture = False
                confirmation_started = False
                confirmation_start = None
                placement_result = None

                message = "TARS: BOX LOCKED"
                status = f"LOCKED: {locked_box_class}"

                print("\n================================")
                print("TARS: BOX LOCKED")
                print("BOX:", locked_box_class)
                print("================================\n")

                speak(f"{color_of(locked_box_class)} box locked.")

            else:
                message = "TARS: NO BOX"
                status = "NO BOX DETECTED"
                print("TARS: No box visible.")
                speak("No box visible.")

    # ========================================================
    # NOT LOCKED
    # ========================================================
    if not locked:
        message = "TARS: DETECT BOX - PRESS L"
        status = "BOX IS NOT LOCKED"

        for detection in visible_boxes:
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                frame,
                f"{detection['class']} {detection['confidence']:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # ========================================================
    # LOCKED
    # ========================================================
    else:
        box_color = color_of(locked_box_class)

        # ----------------------------------------------------
        # Draw Locked box, Placement region, Capture zone
        # ----------------------------------------------------
        lx1, ly1, lx2, ly2 = map(int, locked_box)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 5)
        cv2.putText(frame, f"LOCKED: {locked_box_class}", (lx1, max(30, ly1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        px1, py1, px2, py2 = map(int, placement_box)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(frame.shape[1] - 1, px2), min(frame.shape[0] - 1, py2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 3)
        cv2.putText(frame, "PLACEMENT OUTLINE", (px1, max(25, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cx1, cy1, cx2, cy2 = map(int, capture_box)
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(frame.shape[1] - 1, cx2), min(frame.shape[0] - 1, cy2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        cv2.putText(frame, "CAPTURE ZONE", (cx1, min(frame.shape[0] - 10, cy2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ====================================================
        # RESULT ALREADY AVAILABLE
        # ====================================================
        if placement_result is not None:
            if placement_result == "CORRECT":
                message = f"{ball_color.upper()} BALL SUCCESSFULLY INSERTED"
                status = f"{ball_color} ball -> {box_color} box : CORRECT"
                color = (0, 255, 0)

            elif placement_result == "SKIPPED":
                message = "STEP SKIPPED"
                status = f"Tried {ball_color} in {box_color} early"
                color = (0, 165, 255) # Orange Warning

            else:
                message = "WRONG COLOR / SEQUENCE"
                status = f"{ball_color} ball -> {box_color} box : WRONG"
                color = (0, 0, 255)

            cv2.putText(frame, message, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)
            cv2.putText(frame, "PRESS C TO CLEAR CURRENT LOCK", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ====================================================
        # NO RESULT YET
        # ====================================================
        else:
            ball = None
            if visible_balls:
                ball = max(visible_balls, key=lambda d: d["confidence"])

            # =================================================
            # BALL DETECTED
            # =================================================
            if ball is not None:
                current_ball_box = ball["box"]
                current_ball_class = ball["class"]
                current_ball_center = center(current_ball_box)

                fake_box_region = expand_box(locked_box, BOX_IGNORE_MARGIN)
                looks_like_locked_box = overlap(current_ball_box, fake_box_region)

                if looks_like_locked_box and not ball_started_outside:
                    ball_draw_color = (0, 0, 255)
                else:
                    ball_draw_color = (255, 0, 255)

                bx1, by1, bx2, by2 = current_ball_box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), ball_draw_color, 3)
                cv2.circle(frame, current_ball_center, 7, ball_draw_color, -1)
                cv2.putText(
                    frame,
                    f"{current_ball_class} {ball['confidence']:.2f}",
                    (bx1, max(25, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    ball_draw_color,
                    2
                )

                # ==========================================
                # STATE: BALL HAS NOT STARTED OUTSIDE YET
                # ==========================================
                if not ball_started_outside:
                    if looks_like_locked_box:
                        ball_outside_count = 0
                        message = "IGNORING BOX AS BALL"
                        status = "WAITING FOR REAL BALL"

                    elif not inside(current_ball_center, placement_box):
                        ball_outside_count += 1
                        message = "BALL OUTSIDE"
                        status = f"VALIDATING BALL {ball_outside_count}/{BALL_OUTSIDE_FRAMES}"

                        if ball_outside_count >= BALL_OUTSIDE_FRAMES:
                            ball_started_outside = True
                            ball_color = color_of(current_ball_class)
                            trajectory.clear()
                            trajectory.append(current_ball_center)
                            previous_distance = distance_to_box(current_ball_center, locked_box)
                            approach_count = 0

                            print(f"\nTARS: {ball_color} ball validated from outside.\n")
                            speak(f"{ball_color} ball validated.")
                    else:
                        ball_outside_count = 0
                        message = "BALL STARTED INSIDE"
                        status = "BALL MUST START OUTSIDE"

                # ==========================================
                # STATE: BALL MOVING TOWARD PLACEMENT
                # ==========================================
                elif not ball_crossed_placement:
                    trajectory.append(current_ball_center)
                    if inside(current_ball_center, placement_box):
                        ball_crossed_placement = True
                        previous_distance = distance_to_box(current_ball_center, locked_box)
                        approach_count = 0
                        speak("Ball approaching box.")

                    message = "BALL APPROACHING"
                    status = "PLACEMENT OUTLINE CROSSED"

                # ==========================================
                # STATE: BALL APPROACHING CAPTURE ZONE
                # ==========================================
                elif not ball_reached_capture:
                    trajectory.append(current_ball_center)
                    current_distance = distance_to_box(current_ball_center, locked_box)

                    if previous_distance is None:
                        previous_distance = current_distance

                    if current_distance < previous_distance - MOVEMENT_TOLERANCE:
                        approach_count += 1
                    else:
                        approach_count = max(0, approach_count - 1)

                    previous_distance = current_distance
                    reached_capture_zone = inside(current_ball_center, capture_box)

                    if reached_capture_zone and approach_count >= APPROACH_FRAMES_REQUIRED:
                        ball_reached_capture = True
                        confirmation_started = True
                        confirmation_start = time.time()
                        print("\nTARS: BALL REACHED CAPTURE ZONE\n")
                        speak("Ball reached capture zone, confirming.")
                    else:
                        message = "BALL MOVING TOWARD BOX"
                        status = f"APPROACH {approach_count}/{APPROACH_FRAMES_REQUIRED}"

                # ==========================================
                # STATE: CONFIRM INSERTION
                # ==========================================
                if ball_reached_capture and confirmation_started:
                    elapsed = time.time() - confirmation_start
                    remaining = max(0, CONFIRMATION_TIME - elapsed)
                    message = "BALL REACHED BOX"
                    status = f"CONFIRMING {remaining:.1f}s"

                    if elapsed >= CONFIRMATION_TIME:
                        confirmation_started = False

                        # --- SEQUENCE CHECK LOGIC ---
                        placement_result = check_placement_sequence(ball_color, box_color)

                        if placement_result == "CORRECT":
                            print("TARS: CORRECT PLACEMENT AND SEQUENCE")
                            speak(f"Correct. {ball_color} placed in {box_color}.")

                            # Check if that was the last step
                            if current_step_index >= len(REQUIRED_SEQUENCE):
                                speak("All steps complete. Excellent work.")

                        elif placement_result == "SKIPPED":
                            print("TARS: STEP SKIPPED")
                            speak("Step skipped. Please follow the correct pattern.")
                        else:
                            print("TARS: WRONG COLOR OR SEQUENCE")
                            speak("Wrong color or wrong sequence.")

                        save_event(f"{ball_color} ball", f"{box_color} box", placement_result)

            # =================================================
            # BALL TEMPORARILY NOT DETECTED (OCCLUDED)
            # =================================================
            elif ball_reached_capture and confirmation_started:
                elapsed = time.time() - confirmation_start
                remaining = max(0, CONFIRMATION_TIME - elapsed)
                message = "BALL OCCLUDED"
                status = f"INSERTION CONFIRMING {remaining:.1f}s"

                cv2.putText(frame, "BALL OCCLUDED", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                if elapsed >= CONFIRMATION_TIME:
                    confirmation_started = False

                    # --- SEQUENCE CHECK LOGIC ---
                    placement_result = check_placement_sequence(ball_color, box_color)

                    if placement_result == "CORRECT":
                        print("TARS: CORRECT PLACEMENT AND SEQUENCE")
                        speak(f"Correct. {ball_color} placed in {box_color}.")

                        if current_step_index >= len(REQUIRED_SEQUENCE):
                            speak("All steps complete. Excellent work.")

                    elif placement_result == "SKIPPED":
                        print("TARS: STEP SKIPPED")
                        speak("Step skipped. Please follow the correct pattern.")
                    else:
                        print("TARS: WRONG COLOR OR SEQUENCE")
                        speak("Wrong color or wrong sequence.")

                    save_event(f"{ball_color} ball", f"{box_color} box", placement_result)

            # =================================================
            # WAITING FOR BALL
            # =================================================
            elif not ball_started_outside:
                message = "WAITING FOR BALL"
                status = "BALL MUST START OUTSIDE"
            else:
                message = "BALL TRACKING"
                status = "MOVE BALL TOWARD BOX"


    # ========================================================
    # TOP HUD
    # ========================================================
    cv2.rectangle(frame, (0, 0), (width, 105), (0, 0, 0), -1)
    cv2.putText(frame, "TARS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, message, (110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Display Current Goal Step
    if current_step_index < len(REQUIRED_SEQUENCE):
        req_ball, req_box = REQUIRED_SEQUENCE[current_step_index]
        goal_str = f"GOAL: {req_ball.upper()} -> {req_box.upper()}"
    else:
        goal_str = "GOAL: ALL STEPS COMPLETE"

    cv2.putText(frame, goal_str, (max(20, width - 420), 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # ========================================================
    # KEYBOARD HELP
    # ========================================================
    cv2.putText(
        frame,
        "L:LOCK  C:CLEAR  N:NEXT-STEP  Q:QUIT",
        (max(20, width - 420), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # ========================================================
    # DISPLAY
    # ========================================================
    cv2.imshow("TARS - AI Experiment", frame)

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("\nTARS stopped.")
