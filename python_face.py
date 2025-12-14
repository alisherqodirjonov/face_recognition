"""
Docstring for python_face
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

# ================= FILE PATHS (MUST EXIST) =================
# Download these from links provided above and put in script dir
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
KNOWN_DIR = "./Media/known"
# ===========================================================


# --------------- TUNE THESE ---------------
CAMERA_INDEX = 0
# Models for precision: "Facenet512" (best balance), "ArcFace" (highly precise, slower).
# Models for speed: "VGG-Face" (faster, less precise).
MODEL_NAME = "Facenet512" 
# Threshold depends on model. For Facenet512, 0.3-0.4 is usually good cosine distance.
DIST_THRESHOLD = 0.35      
FRAME_WIDTH = 640            # Processing width. Keep it reasonable (640-800).
DETECT_EVERY = 5             # Run expensive DNN detection every N frames.
RECOG_EVERY = 15             # Re-run deepface recognition on tracked objects every M frames.
MAX_WORKERS = 2              # Background threads for embeddings.
DNN_CONFIDENCE = 0.3         # Minimum confidence for the DNN face detector.
# ------------------------------------------


def load_dnn_detector():
    if not os.path.exists(DNN_PROTO) or not os.path.exists(DNN_MODEL):
        print("\n[ERROR] MISSING DNN MODEL FILES.")
        print("Please download 'deploy.prototxt.txt' and 'res10_300x300_ssd_iter_140000.caffemodel'")
        print("and place them in the script directory.\n")
        exit(1)
    print("[INFO] Loading DNN face detector...")
    net = cv.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    # Optional: If you have an Intel GPU or compatible hardware, enable OpenCL targets for a speed boost.
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) 
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL) 
    return net

def detect_faces_dnn(net, frame, conf_threshold=0.5):
    """Performs accurate face detection using OpenCV DNN module."""
    h, w = frame.shape[:2]
    # DNN model expects 300x300 blob
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    # Loop over detections [0, 0, index, [confidence, x1, y1, x2, y2]]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure BBox is within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            f_w = endX - startX
            f_h = endY - startY
            
            # Filter out impossibly small detections
            if f_w < 20 or f_h < 20:
                continue

            faces.append((startX, startY, f_w, f_h))
    return faces

def make_tracker():
    """Tries to create a KCF tracker (fastest CPU tracker)."""
    try:
        # Try a sequence of available tracker constructors (legacy or main module)
        # Prefer KCF, but fall back to CSRT, MOSSE, MIL if available.
        # Many OpenCV installs require `opencv-contrib-python` to have tracker implementations.
        constructors = []
        if hasattr(cv, 'legacy'):
            # legacy namespace (newer OpenCV with contrib)
            for name in ('TrackerKCF_create', 'TrackerCSRT_create', 'TrackerMOSSE_create', 'TrackerMIL_create'):
                if hasattr(cv.legacy, name):
                    constructors.append(getattr(cv.legacy, name))
        # main module (older API or some builds expose constructors here)
        for name in ('TrackerKCF_create', 'TrackerCSRT_create', 'TrackerMOSSE_create', 'TrackerMIL_create'):
            if hasattr(cv, name):
                constructors.append(getattr(cv, name))

        # Try each constructor until one succeeds
        for ctor in constructors:
            try:
                return ctor()
            except Exception:
                continue

        # Last-ditch attempt: some very old builds expose Tracker_create factory
        if hasattr(cv, 'Tracker_create'):
            try:
                return cv.Tracker_create('KCF')
            except Exception:
                pass

        raise AttributeError("No tracker constructors found in cv2. Install 'opencv-contrib-python' to enable trackers.")
    except Exception as e:
         print(f"[ERROR] Could not create tracker: {e}. Ensure 'opencv-contrib-python' is installed (matching OpenCV version).")
         print("You can install it with: python -m pip install --upgrade opencv-contrib-python")
         exit(1)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 1.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 1.0
    return 1.0 - (np.dot(a, b) / (na * nb))

def load_known_embeddings(known_dir: str, model_name: str):
    embeddings = {}
    if not os.path.exists(known_dir):
         os.makedirs(known_dir)
         print(f"[INFO] Created {known_dir}. Please put images there.")
         exit(1)
         
    files = [f for f in os.listdir(known_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[INFO] Loading {len(files)} known images using {model_name}...")
    
    # We use enforce_detection=True here to ensure we are embedding faces from the known images.
    # Using MTCNN is slow but precise for the initial setup phase.
    for fname in files:
        path = os.path.join(known_dir, fname)
        name = os.path.splitext(fname)[0]
        try:
            # Using a better detector for registration improves overall precision
            rep = DeepFace.represent(img_path=path, model_name=model_name, enforce_detection=True, detector_backend="mtcnn")
            if isinstance(rep, list): rep = rep[0]
            embeddings[name] = np.array(rep['embedding'])
            print(f"  - Loaded: {name}")
        except Exception as e:
            print(f"  ! Failed to load {fname}: {e}") # e.g. face not detected in known image
            
    if not embeddings and files:
         print("[WARN] No faces detected in known images. Check image quality.")
         
    return embeddings

def compute_face_embedding(face_crop: np.ndarray, model_name: str):
    """Computes embedding on a pre-cropped face image."""
    try:
        h, w = face_crop.shape[:2]
        # Ensure the crop isn't tiny or empty before sending to heavy model
        if h < 40 or w < 40: return None

        # enforce_detection=False is CRITICAL for speed here. 
        # We trust the tracker/detector provided a valid face crop.
        rep = DeepFace.represent(face_crop, model_name=model_name, enforce_detection=False)
        
        if isinstance(rep, list): rep = rep[0]
        return np.array(rep['embedding'])
        
    except Exception as e:
        # print(f"[DEBUG] Embedding failed: {e}") # Occasional failures are normal during tracking motion blur
        return None

def match_embedding(emb, known_embs, threshold):
    if not known_embs: return None, None
    best_name = None; best_dist = 1.0
    for n, ke in known_embs.items():
        d = cosine_distance(emb, ke)
        if d < best_dist:
            best_dist = d; best_name = n
    if best_dist <= threshold:
        return best_name, float(best_dist)
    return None, best_dist

# -------------------- Main Pipeline --------------------
def main():
    # 1. Load Resources
    detector_net = load_dnn_detector()
    known = load_known_embeddings(KNOWN_DIR, MODEL_NAME)

    cap = cv.VideoCapture(CAMERA_INDEX)
    # Try setting cam resolution higher, we resize for processing anyway
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened(): raise RuntimeError("Cannot open camera")

    frame_counter = 0
    timer = time.time()
    fps_counter = 0
    FPS = 0

    # Trackers data structure: 
    # tid -> {'tracker': cv object, 'bbox': (x,y,w,h), 'last_recog': int, 'name': str, 'dist': float}
    trackers = {}
    trackers_lock = Lock()
    # The pool handles the heavy lifting of embedding generation
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    print(f"[INFO] Starting recognition using {MODEL_NAME}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_counter += 1

        # 2. Preprocessing (Resize)
        # Working on a smaller frame significantly speeds up detection and tracking
        orig_h, orig_w = frame.shape[:2]
        aspect_ratio = orig_w / orig_h
        process_w = FRAME_WIDTH
        process_h = int(process_w / aspect_ratio)
        
        process_frame = cv.resize(frame, (process_w, process_h))
        scale_x = orig_w / process_w
        scale_y = orig_h / process_h


        # 3. Tracker Update Step (Runs every frame on resized image)
              # 3. Tracker Update Step (Runs every frame on resized image)
        with trackers_lock:
            to_delete = []
            for tid, info in list(trackers.items()):
                # Update tracker on the processed (smaller) frame
                try:
                    ok_bbox = info['tracker'].update(process_frame)
                except Exception as e:
                    # If update raised, drop tracker gracefully
                    print(f"[DEBUG] tracker.update() exception for {tid}: {e}")
                    to_delete.append(tid)
                    continue

                # handle both return conventions robustly
                if isinstance(ok_bbox, tuple) and len(ok_bbox) == 2:
                    ok, bbox_small = ok_bbox
                else:
                    # some implementations may return bbox only (rare) or None
                    # treat this as failure to avoid crashes
                    ok = True
                    bbox_small = ok_bbox

                if not ok or bbox_small is None:
                    to_delete.append(tid)
                    continue

                # Save small bbox to reuse later (avoid calling update() again)
                info['small_bbox'] = bbox_small

                # Scale bbox back to original frame size for display and cropping
                sx, sy, sw, sh = map(int, bbox_small)
                x = int(sx * scale_x)
                y = int(sy * scale_y)
                w = int(sw * scale_x)
                h = int(sh * scale_y)

                # Safety bounds checks
                x = max(0, x); y = max(0, y)
                w = min(orig_w - x, w); h = min(orig_h - y, h)

                info['bbox'] = (x, y, w, h)

                # Periodic Recognition Submission Step (unchanged)
                if frame_counter - info['last_recog'] >= RECOG_EVERY and w > 50 and h > 50:
                    roi = frame[y:y+h, x:x+w].copy()
                    info['last_recog'] = frame_counter

                    def recog_task(tid_local, roi_local):
                        emb = compute_face_embedding(roi_local, MODEL_NAME)
                        name_found = None; dist_found = None
                        if emb is not None:
                            name_found, dist_found = match_embedding(emb, known, DIST_THRESHOLD)

                        with trackers_lock:
                            if tid_local in trackers:
                                trackers[tid_local]['name'] = name_found
                                trackers[tid_local]['dist'] = dist_found

                    executor.submit(recog_task, tid, roi)

            for tid in to_delete:
                del trackers[tid]


        # 5. Periodic Detection Step (Runs rarely)
        if frame_counter % DETECT_EVERY == 0:
            detected_faces_small = detect_faces_dnn(detector_net, process_frame, DNN_CONFIDENCE)

            with trackers_lock:
                for (sx, sy, sw, sh) in detected_faces_small:
                    matched = False
                    sx2, sy2 = sx + sw, sy + sh

                    for tid, info in list(trackers.items()):
                        # use last small_bbox from the update pass (safer)
                        bbox_small = info.get('small_bbox')
                        if bbox_small is None:
                            # if we don't have a stored small bbox, skip this tracker for now
                            continue

                        tx, ty, tw, th = map(int, bbox_small)
                        tx2, ty2 = tx + tw, ty + th

                        # Calculate IoU (Intersection over Union)
                        ix1 = max(sx, tx); iy1 = max(sy, ty)
                        ix2 = min(sx2, tx2); iy2 = min(sy2, ty2)
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                        box_area = sw * sh
                        tracker_area = tw * th
                        union_area = box_area + tracker_area - inter_area

                        if union_area > 0 and (inter_area / union_area) > 0.35:
                            matched = True
                            break

                    if matched:
                        continue

                    # New face initialization (same as before)
                    tracker = make_tracker()
                    try:
                        ok = tracker.init(process_frame, (sx, sy, sw, sh))
                    except Exception as e:
                        print(f"[DEBUG] tracker.init exception: {e}")
                        ok = False

                    if not ok:
                        continue

                    tid = str(uuid.uuid4())[:8]

                    x_big = int(sx * scale_x); y_big = int(sy * scale_y)
                    w_big = int(sw * scale_x); h_big = int(sh * scale_y)

                    trackers[tid] = {
                        'tracker': tracker,
                        'bbox': (x_big, y_big, w_big, h_big),
                        'small_bbox': (sx, sy, sw, sh),  # store initial small bbox
                        'last_recog': frame_counter - RECOG_EVERY,
                        'name': None, 'dist': None
                    }



        # 7. Drawing & Display Step
        display_frame = frame.copy()
        fps_counter += 1
        now = time.time()
        if now - timer > 1.0:
            FPS = fps_counter / (now - timer)
            fps_counter = 0
            timer = now

        with trackers_lock:
            for tid, info in trackers.items():
                x, y, w, h = info['bbox']
                name = info.get('name')
                dist = info.get('dist')
                
                color = (0, 255, 0) if name else (0, 0, 255)
                label = f"{name} ({dist:.2f})" if name else "Unknown"
                if dist and name is None: label += f" ({dist:.2f})" # Show dist even if unknown for debugging

                cv.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                # Draw filled box for text background for better readability
                (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv.rectangle(display_frame, (x, y - 20), (x + tw + 4, y), color, -1)
                cv.putText(display_frame, label, (x + 2, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv.putText(display_frame, f"FPS: {int(FPS)} | Model: {MODEL_NAME}", (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        
        # Resize for final display if screen is small
        final_h, final_w = display_frame.shape[:2]
        if final_w > 1000:
             scale_disp = 1000 / final_w
             display_frame = cv.resize(display_frame, (0,0), fx=scale_disp, fy=scale_disp)
             
        cv.imshow("Precise Multi-Face Tracking", display_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Shutting down...")
    executor.shutdown(wait=False)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    