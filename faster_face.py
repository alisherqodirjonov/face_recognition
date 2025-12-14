"""
Faster multi-face recognition using tracking + occasional recognition

Key ideas:
 - Detect every DETECT_EVERY frames (cheap when done on a resized frame)
 - Create a lightweight tracker (KCF) for each detected face
 - Compute deep embeddings only on tracker creation and every RECOG_EVERY frames
 - Use a ThreadPoolExecutor for embedding computation so main loop isn't blocked

Requirements:
 - deepface
 - opencv-python
"""

import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import uuid

import cv2 as cv
import numpy as np
from deepface import DeepFace


# helper to create an OpenCV tracker with multiple fallbacks
def make_tracker(preferred='KCF'):
    """Return a tracker instance. Tries several constructors for compatibility
    with different OpenCV builds (with/without contrib / legacy namespace).
    Raises RuntimeError if no constructor is available.
    """
    # try specific implementations first (KCF is fast/good)
    constructors = []
    if preferred == 'KCF':
        constructors = [
            lambda: cv.TrackerKCF_create(),
            lambda: cv.legacy.TrackerKCF_create(),
        ]
    # also try CSRT / MOSSE as fallbacks
    constructors.extend([
        lambda: cv.TrackerCSRT_create(),
        lambda: cv.legacy.TrackerCSRT_create(),
        lambda: cv.TrackerMOSSE_create(),
        lambda: cv.legacy.TrackerMOSSE_create(),
    ])

    for ctor in constructors:
        try:
            return ctor()
        except Exception:
            continue

    # try generic factory if available
    try:
        return cv.Tracker_create(preferred)
    except Exception:
        pass

    raise RuntimeError("No compatible OpenCV tracker constructor found. Install 'opencv-contrib-python' or use a different OpenCV build.")

# --------------- TUNE THESE ---------------
KNOWN_DIR = "./Media/known"
CAMERA_INDEX = 0
MODEL_NAME = "VGG-Face"       # try "Facenet" or "ArcFace" if you have GPU or need accuracy
DIST_THRESHOLD = 0.40        # lower = stricter
FRAME_WIDTH = 640            # detect on a smaller width -> faster
DETECT_EVERY = 6             # run face detection every N frames
RECOG_EVERY = 10             # re-run embedding/recognition for a given tracked face every M frames
MAX_WORKERS = 2              # threadpool for embeddings
ENFORCE_DETECTION = False    # False is faster and tolerates smaller crops
# ------------------------------------------

# simple cosine distance
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 1.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 1.0
    return 1.0 - (np.dot(a, b) / (na * nb))

# Load known embeddings once (same as previous version)
def load_known_embeddings(known_dir: str, model_name: str):
    embeddings = {}
    files = [f for f in os.listdir(known_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        raise RuntimeError(f"No images found in {known_dir}.")
    print(f"[INFO] Loading {len(files)} known images...")
    for fname in files:
        path = os.path.join(known_dir, fname)
        name = os.path.splitext(fname)[0]
        try:
            rep = DeepFace.represent(img_path=path, model_name=model_name, enforce_detection=ENFORCE_DETECTION)
            if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
                vec = np.array(rep[0]['embedding'])
            elif isinstance(rep, dict) and 'embedding' in rep:
                vec = np.array(rep['embedding'])
            else:
                vec = np.array(rep)
            embeddings[name] = vec
            print(f"  - {name}")
        except Exception as e:
            print(f"  ! failed to load {fname}: {e}")
    if not embeddings:
        raise RuntimeError("No known embeddings computed.")
    return embeddings

# compute embedding for ROI (image array)
def compute_face_embedding(face_img: np.ndarray, model_name: str):
    try:
        h, w = face_img.shape[:2]
        if max(h, w) < 64:
            scale = 128 / max(h, w)
            face_img = cv.resize(face_img, (0, 0), fx=scale, fy=scale)
        rep = DeepFace.represent(face_img, model_name=model_name, enforce_detection=ENFORCE_DETECTION)
        if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
            vec = np.array(rep[0]['embedding'])
        elif isinstance(rep, dict) and 'embedding' in rep:
            vec = np.array(rep['embedding'])
        else:
            vec = np.array(rep)
        return vec
    except Exception as e:
        # avoid crash; log and return None
        print(f"[WARN] embedding failed: {e}")
        return None

# match to known
def match_embedding(emb, known_embs, threshold):
    best_name = None; best_dist = 1.0
    for n, ke in known_embs.items():
        d = cosine_distance(emb, ke)
        if d < best_dist:
            best_dist = d; best_name = n
    if best_dist <= threshold:
        return best_name, float(best_dist)
    return None, None

# -------------------- Main faster pipeline --------------------
def main():
    known = load_known_embeddings(KNOWN_DIR, MODEL_NAME)

    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): raise RuntimeError("Cannot open camera")

    # cascade (cheap) for detection on resized frames
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_counter = 0
    timer = time.time()
    fps_counter = 0
    FPS = 0

    # trackers: id -> dict {tracker, bbox, last_recog_frame, name, dist}
    trackers = {}
    trackers_lock = Lock()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    print("[INFO] Faster face recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_counter += 1
        orig_h, orig_w = frame.shape[:2]

        # resize for detection to speed up
        scale = FRAME_WIDTH / float(orig_w)
        small = cv.resize(frame, (FRAME_WIDTH, int(orig_h * scale)))
        small_gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

        # update trackers each frame (cheap)
        with trackers_lock:
            to_delete = []
            for tid, info in trackers.items():
                ok, bbox = info['tracker'].update(frame)  # trackers operate on full-res frame
                if not ok:
                    # lost tracker
                    to_delete.append(tid)
                    continue
                # save updated bbox
                x, y, w, h = map(int, bbox)
                info['bbox'] = (x, y, w, h)
                info['age'] += 1

                # periodically re-run recognition in background
                if frame_counter - info['last_recog_frame'] >= RECOG_EVERY:
                    # capture ROI copy (small copy) and submit
                    x1, y1, w1, h1 = info['bbox']
                    # safety bounds
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1], x1 + w1); y2 = min(frame.shape[0], y1 + h1)
                    roi = frame[y1:y2, x1:x2].copy()
                    info['last_recog_frame'] = frame_counter
                    # submit embedding job
                    def recog_task(tid_local, roi_local):
                        emb = compute_face_embedding(roi_local, MODEL_NAME)
                        if emb is not None:
                            name, dist = match_embedding(emb, known, DIST_THRESHOLD)
                        else:
                            name, dist = None, None
                        with trackers_lock:
                            if tid_local in trackers:
                                trackers[tid_local]['name'] = name
                                trackers[tid_local]['dist'] = dist
                    executor.submit(recog_task, tid, roi)

            for tid in to_delete:
                del trackers[tid]

        # detection only every DETECT_EVERY frames
        if frame_counter % DETECT_EVERY == 0:
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # map small coords back to original
            detected_boxes = []
            for (sx, sy, sw, sh) in faces:
                x = int(sx / scale); y = int(sy / scale); w = int(sw / scale); h = int(sh / scale)
                # convert to full-res bbox
                detected_boxes.append((x, y, w, h))

            # simple matching: for each detected box, see if it overlaps an existing tracker; if not -> create tracker
            with trackers_lock:
                for (x, y, w, h) in detected_boxes:
                    # compute IoU-ish overlap with existing
                    matched = False
                    for tid, info in trackers.items():
                        tx, ty, tw, th = info['bbox']
                        # overlap check
                        ix = max(0, min(x + w, tx + tw) - max(x, tx))
                        iy = max(0, min(y + h, ty + th) - max(y, ty))
                        inter = ix * iy
                        union = (w * h) + (tw * th) - inter
                        if union > 0 and (inter / union) > 0.3:
                            matched = True
                            break
                    if matched:
                        continue
                    # new face -> create tracker and schedule immediate recognition
                    # create tracker (try preferred KCF, fall back to others)
                    try:
                        tracker = make_tracker('KCF')
                    except RuntimeError as e:
                        print(f"[ERROR] tracker creation failed: {e}")
                        continue

                    bbox = (x, y, w, h)
                    ok = tracker.init(frame, bbox)
                    if not ok:
                        continue
                    tid = str(uuid.uuid4())[:8]
                    trackers[tid] = {
                        'tracker': tracker,
                        'bbox': bbox,
                        'last_recog_frame': frame_counter - RECOG_EVERY,  # force immediate recog
                        'name': None,
                        'dist': None,
                        'age': 0
                    }
                    # schedule immediate recog
                    roi = frame[y:y+h, x:x+w].copy()
                    def init_recog(tid_local, roi_local):
                        emb = compute_face_embedding(roi_local, MODEL_NAME)
                        if emb is not None:
                            name, dist = match_embedding(emb, known, DIST_THRESHOLD)
                        else:
                            name, dist = None, None
                        with trackers_lock:
                            if tid_local in trackers:
                                trackers[tid_local]['name'] = name
                                trackers[tid_local]['dist'] = dist
                    executor.submit(init_recog, tid, roi)

        # draw tracking results
        display_frame = frame.copy()
        with trackers_lock:
            for tid, info in trackers.items():
                x, y, w, h = info['bbox']
                name = info.get('name')
                dist = info.get('dist')
                if name:
                    label = f"{name} {dist:.2f}"
                    cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(display_frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(display_frame, "Unknown", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # status + timestamp + FPS
        fps_counter += 1
        now = time.time()
        if now - timer > 1.0:
            FPS = fps_counter / (now - timer)
            fps_counter = 0
            timer = now
        cv.putText(display_frame, f"FPS:{int(FPS)}", (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
        cv.putText(display_frame, f"{datetime.datetime.utcnow()}", (10, display_frame.shape[0]-10),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)

        # show smaller for display
        h, w = display_frame.shape[:2]
        scale_display = 920 / float(w)
        disp = cv.resize(display_frame, (920, int(h * scale_display)))
        cv.imshow("Fast Multi Recognition", disp)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown(wait=True)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
