"""
Multi-person real-time face recognition (uses DeepFace embeddings)

Place known/reference images in ./Media/known/
File names (without extension) will be used as person names.

Author: alisharify (adapted)
"""

import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2 as cv
import numpy as np
from deepface import DeepFace

# -------------------- User settings --------------------
KNOWN_DIR = "./Media/known"    # directory with known faces (jpg/png). filename -> name
CAMERA_INDEX = 0               # camera id (0 default)
MODEL_NAME = "VGG-Face"        # DeepFace model to use for embeddings (VGG-Face, Facenet, ArcFace, etc.)
DIST_THRESHOLD = 0.40          # cosine distance threshold (lower = stricter). Tune as needed.
ENFORCE_DETECTION = False      # use False to be tolerant of small/partial faces
MAX_WORKERS = 2                # threadpool size for computing embeddings per-frame (1-4 typical)
FRAME_RESIZE_WIDTH = 920       # final window width (visual only)
# -------------------------------------------------------

# load cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# helper: compute cosine distance between two vectors
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # if either vector zero-length, return large distance
    if a is None or b is None:
        return 1.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# load known images and compute embeddings once
def load_known_embeddings(known_dir: str, model_name: str):
    embeddings = {}   # name -> embedding (np.array)
    files = [f for f in os.listdir(known_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        raise RuntimeError(f"No images found in {known_dir}. Put known images there (jpg/png).")
    print(f"[INFO] Loading {len(files)} known images from {known_dir} ...")
    for fname in files:
        path = os.path.join(known_dir, fname)
        name = os.path.splitext(fname)[0]
        try:
            # DeepFace.represent returns a list of dicts when dataset param is None for some versions,
            # but for single img it's usually a vector (list/np.array). We normalize into np.array.
            rep = DeepFace.represent(img_path=path, model_name=model_name, enforce_detection=ENFORCE_DETECTION)
            # rep might be list or dict depending on version. Try to extract vector robustly:
            if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
                vec = np.array(rep[0]['embedding'])
            elif isinstance(rep, dict) and 'embedding' in rep:
                vec = np.array(rep['embedding'])
            else:
                # rep may be a plain list of floats
                vec = np.array(rep)
            embeddings[name] = vec
            print(f"  - Loaded: {name}")
        except Exception as e:
            print(f"  ! Failed to process {fname}: {e}")
    if not embeddings:
        raise RuntimeError("No embeddings were computed for known images.")
    return embeddings

# compute embedding for face ROI (image array)
def compute_face_embedding(face_img: np.ndarray, model_name: str):
    try:
        # resize ROI to reasonable size for DeepFace
        h, w = face_img.shape[:2]
        if max(h, w) < 64:
            # scale up small faces to help representation
            scale = 128 / max(h, w)
            face_img = cv.resize(face_img, (0, 0), fx=scale, fy=scale)
        # DeepFace.represent can accept an image array
        rep = DeepFace.represent(face_img, model_name=model_name, enforce_detection=ENFORCE_DETECTION)
        if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
            vec = np.array(rep[0]['embedding'])
        elif isinstance(rep, dict) and 'embedding' in rep:
            vec = np.array(rep['embedding'])
        else:
            vec = np.array(rep)
        return vec
    except Exception as e:
        # print for debugging but don't crash
        print(f"[WARN] embedding failed: {e}")
        return None

# match embedding to known embeddings -> (best_name, best_distance) or (None, None)
def match_embedding(emb: np.ndarray, known_embeddings: dict, threshold: float):
    best_name = None
    best_dist = 1.0
    for name, known_emb in known_embeddings.items():
        d = cosine_distance(emb, known_emb)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= threshold:
        return best_name, float(best_dist)
    return None, None

# -------------------- Main --------------------
def main():
    # load known embeddings
    known_embeddings = load_known_embeddings(KNOWN_DIR, MODEL_NAME)

    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    timer = time.time()
    imageCounter = 0
    FPS = 0
    lock = Lock()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    print("[INFO] Starting camera. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # keep a copy for processing
        proc_frame = frame.copy()
        gray = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)

        # detect faces (x, y, w, h)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []  # will hold dicts for each face: {'rect': (x,y,w,h), 'name': name or None, 'dist': float or None}

        # For each detected face, compute embedding in thread pool (to utilize CPU while cascade continues)
        futures = []
        for (x, y, w, h) in faces:
            # expand the bbox a bit to include hair/face border
            pad_w = int(0.15 * w)
            pad_h = int(0.15 * h)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(proc_frame.shape[1], x + w + pad_w)
            y2 = min(proc_frame.shape[0], y + h + pad_h)
            face_roi = proc_frame[y1:y2, x1:x2]
            # submit embedding computation
            futures.append(((x1, y1, x2 - x1, y2 - y1), executor.submit(compute_face_embedding, face_roi, MODEL_NAME)))

        # collect futures
        for (rect, fut) in futures:
            emb = fut.result(timeout=5)  # wait (short) for embedding
            if emb is None:
                results.append({'rect': rect, 'name': None, 'dist': None})
            else:
                name, dist = match_embedding(emb, known_embeddings, DIST_THRESHOLD)
                results.append({'rect': rect, 'name': name, 'dist': dist})

        # Draw results on frame
        face_found = any(r['name'] is not None for r in results)
        for r in results:
            x, y, w, h = r['rect']
            if r['name']:
                # matched
                label = f"{r['name']} ({r['dist']:.2f})"
                # green box for matched
                cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # unknown face -> red box
                cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
                cv.putText(frame, "Unknown", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # top-left status
        if face_found:
            status_text = "Face Match"
            status_color = (0, 0, 255)
        else:
            status_text = "No Match"
            status_color = (0, 255, 0)

        cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)

        # add timestamp
        cv.putText(frame, f"{datetime.datetime.utcnow()}", (10, frame.shape[0] - 10),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # FPS counting
        imageCounter += 1
        now = time.time()
        if now - timer > 1:
            FPS = imageCounter / (now - timer)
            imageCounter = 0
            timer = now
        cv.putText(frame, f"FPS:{int(FPS)}", (frame.shape[1] - 110, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # resize display window
        h, w = frame.shape[:2]
        scale = FRAME_RESIZE_WIDTH / float(w)
        display = cv.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))

        cv.imshow("Multi Recognition", display)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown(wait=True)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
