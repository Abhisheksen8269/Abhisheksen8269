import cv2
import numpy as np
import subprocess
import signal
import sys
import os

# === Load Cascade Classifiers ===
cascade_dir = '/opt/edgeai-gst-apps/docker'

face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_smile.xml'))

if face_cascade.empty() or eye_cascade.empty():
    print("❌ Face or eye cascade is missing or corrupted.")
    sys.exit(1)

if mouth_cascade.empty():
    print("⚠️ Mouth cascade is corrupted or incompatible. Disabling mouth detection.")
    mouth_cascade = None

# === Open IMX219 Camera via TI ISP ===
cap_imx = cv2.VideoCapture(
    "v4l2src device=/dev/video2 io-mode=5 ! "
    "video/x-bayer,format=rggb,width=1920,height=1080 ! "
    "tiovxisp sink_0::device=/dev/v4l-subdev2 sensor-name=SENSOR_SONY_IMX219_RPI "
    "dcc-isp-file=/opt/imaging/imx219/dcc_viss.bin "
    "sink_0::dcc-2a-file=/opt/imaging/imx219/dcc_2a.bin format-msb=7 ! "
    "tiovxmultiscaler ! video/x-raw,format=NV12,width=640,height=480 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1",
    cv2.CAP_GSTREAMER
)

# === USB Camera ===
cap_usb = cv2.VideoCapture("/dev/video18", cv2.CAP_V4L2)
cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap_imx.isOpened() or not cap_usb.isOpened():
    print("❌ One or both cameras failed to open.")
    sys.exit(1)

# === HDMI GStreamer Output (BGR fix applied) ===
gst_cmd = [
    'gst-launch-1.0',
    'fdsrc', '!', 'rawvideoparse',
    'format=bgr', 'width=1280', 'height=480', 'framerate=30/1', '!',
    'videoconvert', '!', 'autovideosink', 'sync=false'
]

proc = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE)

def stop(sig, frame):
    print("\n🛑 Stopping...")
    cap_imx.release()
    cap_usb.release()
    proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

print("▶ Running accurate face/eye/mouth detection...")

last_face = None
no_eye_counter = 0
yawn_counter = 0

while True:
    ret1, imx_frame = cap_imx.read()
    ret2, usb_frame = cap_usb.read()

    if not ret1 or not ret2:
        continue

    gray = cv2.cvtColor(imx_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        face = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = face
        last_face = face

        cv2.rectangle(imx_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imx_frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = imx_frame[y:y + h, x:x + w]

        # === Eye Detection ===
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
        if len(eyes) < 2:
            no_eye_counter += 1
        else:
            no_eye_counter = 0
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        if no_eye_counter > 5:
            cv2.putText(imx_frame, "😴 Sleepy", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # === Mouth Detection ===
        if mouth_cascade:
            mouth_gray = roi_gray[int(h * 0.6):, :]
            mouths = mouth_cascade.detectMultiScale(mouth_gray, scaleFactor=1.03, minNeighbors=8)

            if len(mouths) > 0:
                yawn_counter += 1
                for (mx, my, mw, mh) in mouths[:1]:
                    abs_mx = x + mx
                    abs_my = y + int(h * 0.6) + my
                    cv2.rectangle(imx_frame, (abs_mx, abs_my), (abs_mx + mw, abs_my + mh), (0, 0, 255), 1)
                    cv2.circle(imx_frame, (abs_mx + mw // 2, abs_my + mh // 2), 5, (0, 0, 255), -1)
            else:
                yawn_counter = 0

            if yawn_counter > 3:
                cv2.putText(imx_frame, "😮 Yawning", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    elif last_face is not None:
        x, y, w, h = last_face
        cv2.rectangle(imx_frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.putText(imx_frame, "Tracking...", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # === Combine Feeds ===
    combined = np.hstack((imx_frame, usb_frame))

    try:
        proc.stdin.write(combined.tobytes())
    except Exception as e:
        print("⚠️ GStreamer write error:", e)
        continue

