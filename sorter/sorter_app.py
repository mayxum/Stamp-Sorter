# Stamp sorter with a live web page.
#
# On the Pi (system Python, which has picamera2 and gpiozero):
#     python3 ~/sorter/sorter_app.py
# then open http://thinh.local:8000 on the Mac (or http://<Pi IP>:8000 on the hotspot).
#
# The camera and GPIO run here; classification runs in classifier_worker.py inside the
# CLIP environment (CLIP_PYTHON), so the two Python setups never have to be mixed.
#
# Simulation on the Mac (no Pi hardware; uses photos from a folder and the real classifier):
#     python sorter_app.py --sim --photos ~/Downloads/sim_photos

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# camera / trapdoor region
ROI = (1346, 819, 959, 1236)  # (x, y, w, h) of the trapdoor in full-resolution pixels.
FULL_SIZE = (4608, 2592)
LORES_SIZE = (640, 360)  # small stream used to watch the trapdoor
EXPOSURE_US = 10000  # 1/100 s, confirmed sharp on Nov 6
EV = -1.0

# automatic detection (tune on the real rig; live values show at the bottom of the page)
PIXEL_DIFF = 25  # this much brighter than the empty plate = "something there"
# fraction of the box that must get brighter (stamps 20%+, empty plate ~2%)
PRESENCE_FRAC = 0.10
# frame-to-frame change below this = nothing moving (camera noise ~4 in dim light)
STILL_MOTION = 6.0
STILL_TIME = 0.5  # seconds the stamp must sit still before the photo
EMPTY_MAX_BRIGHT = 100  # empty plate averages ~65; brighter = something is on it
CLEAR_TIMEOUT = 5.0  # seconds after the drop for the trapdoor to read empty

# carousel / trapdoor (same as sorter.py)
STEP_PIN, DIR_PIN, EN_PIN, HOME_PIN, SERVO_PIN = 17, 27, 22, 23, 18
BINS = 5
STEPS_PER_BIN = 40
STEP_DELAY = 0.005
HOMING_DELAY = 0.01
MAX_HOMING_STEPS = 400
# steps after the flag blocks the sensor (negative = back); found with home_offset.py
HOME_OFFSET_STEPS = -10
REHOME_EVERY = 10
BIN_NAMES = ["nature", "people", "places", "graphic", "reject"]
CLOSED_ANGLE, OPEN_ANGLE = 90, 180
OPEN_TIME, SETTLE_TIME = 0.65, 0.5
OPEN_SWEEP = 0.3  # seconds to tilt the trapdoor open; 0 = snap open (the original)

# software
CLIP_PYTHON = os.path.expanduser("~/clip-env/bin/python")
PORT = 8000
PHOTO_DIR = HERE / "photos"
HISTORY = 8


# Hardware
class Hardware:
    def __init__(self):
        from gpiozero import AngularServo, DigitalInputDevice, OutputDevice

        self.step = OutputDevice(STEP_PIN)
        self.direction = OutputDevice(DIR_PIN)
        self.enable = OutputDevice(EN_PIN, active_high=False, initial_value=False)
        self.home_in = DigitalInputDevice(HOME_PIN, pull_up=None, active_state=True)
        self.servo = AngularServo(
            SERVO_PIN,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
            initial_angle=CLOSED_ANGLE,
        )
        time.sleep(0.5)
        self.servo.detach()
        self.current = 0

    def _pulse(self, count, delay):
        for _ in range(count):
            self.step.on()
            time.sleep(delay)
            self.step.off()
            time.sleep(delay)

    def home(self):
        self.enable.on()
        self.direction.on()
        steps = 0
        while self.home_in.value:
            self._pulse(1, HOMING_DELAY)
            steps += 1
            if steps >= MAX_HOMING_STEPS:
                raise RuntimeError("Home sensor stays blocked")
        while not self.home_in.value:
            self._pulse(1, HOMING_DELAY)
            steps += 1
            if steps >= MAX_HOMING_STEPS:
                raise RuntimeError("Home not found after 2 turns")
        if HOME_OFFSET_STEPS < 0:  # negative offset: step back after homing
            self.direction.off()
        self._pulse(abs(HOME_OFFSET_STEPS), HOMING_DELAY)
        self.current = 0
        time.sleep(0.2)
        self.enable.off()

    def rotate_to(self, target):
        diff = (target - self.current) % BINS
        if diff:
            self.enable.on()
            time.sleep(0.05)  # power the motor only while it moves
            if diff <= 2:
                self.direction.on()
                steps = diff * STEPS_PER_BIN
            else:
                self.direction.off()
                steps = (BINS - diff) * STEPS_PER_BIN
            self._pulse(steps, STEP_DELAY)
            time.sleep(0.2)
            self.enable.off()  # idle: no current, motor stays cool
        self.current = target

    def drop(self):
        # a few big steps; tiny ones make the servo stutter
        steps = max(1, round(OPEN_SWEEP / 0.05))
        for i in range(1, steps + 1):  # tilt open gradually
            self.servo.angle = CLOSED_ANGLE + (OPEN_ANGLE - CLOSED_ANGLE) * i / steps
            time.sleep(OPEN_SWEEP / steps)
        time.sleep(OPEN_TIME)
        self.servo.angle = CLOSED_ANGLE
        time.sleep(SETTLE_TIME)
        self.servo.detach()

    def close(self):
        try:
            self.servo.angle = CLOSED_ANGLE
            time.sleep(SETTLE_TIME)
            self.servo.detach()
        finally:
            self.enable.off()
            for device in (self.step, self.direction, self.enable, self.home_in, self.servo):
                device.close()


class SimHardware:
    def __init__(self):
        self.current = 0
        self.on_drop = None

    def home(self):
        time.sleep(1.0)
        self.current = 0

    def rotate_to(self, target):
        diff = (target - self.current) % BINS
        time.sleep(0.4 * min(diff, BINS - diff))
        self.current = target

    def drop(self):
        time.sleep(OPEN_SWEEP + OPEN_TIME + SETTLE_TIME)
        if self.on_drop:
            self.on_drop()

    def close(self):
        pass


# Camera
def lores_roi():
    if not ROI:
        return None
    scale_x, scale_y = LORES_SIZE[0] / FULL_SIZE[0], LORES_SIZE[1] / FULL_SIZE[1]
    x, y, width, height = ROI
    return (
        int(x * scale_x),
        int(y * scale_y),
        max(1, int(width * scale_x)),
        max(1, int(height * scale_y)),
    )


class Camera:
    def __init__(self):
        from libcamera import controls
        from picamera2 import Picamera2

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": FULL_SIZE, "format": "RGB888"},
            lores={"size": LORES_SIZE, "format": "YUV420"},
            buffer_count=3,
        )
        self.cam.configure(config)
        self.cam.set_controls(
            {
                "ExposureTime": EXPOSURE_US,
                "ExposureValue": EV,
                "AfMode": controls.AfModeEnum.Auto,
                "AfRange": controls.AfRangeEnum.Full,
            }
        )
        self.cam.start()
        time.sleep(1.0)
        try:  # lock brightness, so the empty-trapdoor photo and later checks always match
            metadata = self.cam.capture_metadata()
            self.cam.set_controls(
                {"AnalogueGain": metadata["AnalogueGain"], "ColourGains": metadata["ColourGains"]}
            )
            time.sleep(0.3)
            print("Camera brightness locked: gain %.2f" % metadata["AnalogueGain"])
        except Exception as error:
            print("Could not lock camera brightness:", error)

    def gray(self):
        # Small grayscale frame of the trapdoor region (whole frame if ROI is not set).
        luminance = self.cam.capture_array("lores")[: LORES_SIZE[1], : LORES_SIZE[0]].astype(
            np.float32
        )
        region = lores_roi()
        gray_frame = (
            luminance[region[1] : region[1] + region[3], region[0] : region[0] + region[2]]
            if region
            else luminance
        )
        # Average 2x2 blocks: halves the pixel noise, which in a dim room was enough to
        # make an empty plate read as "something there".
        height, width = (gray_frame.shape[0] // 2) * 2, (gray_frame.shape[1] // 2) * 2
        return gray_frame[:height, :width].reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    def capture(self, prefix="stamp"):
        PHOTO_DIR.mkdir(exist_ok=True)
        path = PHOTO_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        self.cam.autofocus_cycle()
        request = self.cam.capture_request()
        try:
            request.save("main", str(path))
        finally:
            request.release()
        return path

    def close(self):
        self.cam.stop()


class SimCamera:
    # Pretends a stamp arrives a few seconds after the trapdoor is empty.

    def __init__(self, folder):
        from PIL import Image

        self.Image = Image
        extensions = (".jpg", ".jpeg", ".png")
        self.photos = sorted(
            path
            for path in Path(folder).expanduser().iterdir()
            if path.suffix.lower() in extensions
        )
        if not self.photos:
            raise SystemExit(f"No photos in {folder}")
        self.index = -1
        self.loaded = None
        self.occupied_at = time.time() + 4.0
        self.occupied = False

    def dropped(self):
        self.occupied = False
        self.occupied_at = time.time() + 4.0

    def gray(self):
        if not self.occupied and time.time() >= self.occupied_at:
            self.occupied = True
            self.index = (self.index + 1) % len(self.photos)
            image = self.Image.open(self.photos[self.index]).convert("L").resize(LORES_SIZE)
            self.loaded = np.asarray(image, dtype=np.float32)
        if self.occupied:
            return self.loaded
        return (
            np.random.default_rng().normal(8, 1, (LORES_SIZE[1], LORES_SIZE[0])).astype(np.float32)
        )

    def capture(self, prefix="stamp"):
        return self.photos[max(self.index, 0)]

    def close(self):
        pass


# Classifier worker
class Worker:
    def __init__(self, python, script):
        self.proc = subprocess.Popen(
            [python, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = None
        self.lock = threading.Lock()

    def wait_ready(self):
        response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Classifier failed to start (see the terminal for the error)")
        self.ready = json.loads(response_line)
        return self.ready

    def request(self, message):
        with self.lock:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
            response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Classifier stopped (see the terminal for the error)")
        return json.loads(response_line)

    def classify(self, photo, roi):
        return self.request({"photo": str(photo), "roi": list(roi) if roi else None})

    def set_empty(self, photo, roi):
        return self.request({"empty": str(photo), "roi": list(roi) if roi else None})

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


# Sorter state machine
class Sorter:
    def __init__(self, hw, cam, worker, sim, use_home=True):
        self.hw, self.cam, self.worker, self.sim = hw, cam, worker, sim
        self.use_home = use_home
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sort_now = threading.Event()
        self.quit = threading.Event()  # set by the Turn off button
        self.recal = threading.Event()  # set by Reset: retake the empty-trapdoor photo
        self.next_id = 0
        self.auto_allowed = sim or ROI is not None
        self.state = {
            "status": "starting",
            "message": "Starting up",
            "auto": self.auto_allowed,
            "auto_allowed": self.auto_allowed,
            "last": None,
            "history": [],
            "counts": [0] * BINS,
            "classes": BIN_NAMES[:4],
            "threshold": 0.70,
            "metrics": {},
            "cycles": 0,
            "sim": sim,
        }
        self.base = None

    # Shared state
    def set(self, status, message):
        with self.lock:
            self.state["status"], self.state["message"] = status, message

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def set_auto(self, on):
        with self.lock:
            self.state["auto"] = bool(on) and self.auto_allowed

    def reset(self):
        # Clear the counts and recent stamps on the page. Saved crops are kept.
        with self.lock:
            self.state.update({"last": None, "history": [], "counts": [0] * BINS, "cycles": 0})
        self.recal.set()

    # Trapdoor detection and calibration
    def measure(self, prev):
        frame = self.cam.gray()
        # Only pixels BRIGHTER than the empty plate count: stamps are light paper on a dark plate,
        # while shadows, a slightly shifted plate or the dark gap only make things darker.
        presence = (
            float(np.mean((frame - self.base) > PIXEL_DIFF)) if self.base is not None else 0.0
        )
        motion = (
            float(np.mean(np.abs(frame - prev)))
            if prev is not None and prev.shape == frame.shape
            else 99.0
        )
        with self.lock:
            self.state["metrics"] = {
                "presence": round(100 * presence, 1),
                "motion": round(motion, 2),
            }
        return frame, presence, motion

    def calibrate(self):
        # Take the empty-trapdoor photo. Refuses while a stamp is on the plate (it is dark when empty).
        while not self.stop.is_set():
            self.set("calibrating", "Keep the trapdoor empty")
            frames = []
            for _ in range(10):
                frames.append(self.cam.gray())
                time.sleep(0.1)
            base = np.mean(frames, axis=0)
            if ROI is None or self.sim or float(base.mean()) < EMPTY_MAX_BRIGHT:
                self.base = base
                return
            self.set("stuck", "Trapdoor is not empty. Remove the stamp")
            time.sleep(1.0)

    def fingerprint(self):
        # Show the classifier the empty trapdoor, so it can recognise it later.
        if ROI is None or self.sim:
            return
        self.set("calibrating", "Learning what the empty trapdoor looks like")
        reply = self.worker.set_empty(self.cam.capture("empty"), ROI)
        if not reply.get("ok"):
            print("Empty-trapdoor fingerprint failed:", reply.get("error"))

    def plate_is_empty(self):
        # Photo + classifier: is the trapdoor really empty, even if the pixel check disagrees?
        reply = self.worker.classify(self.cam.capture("check"), ROI)
        return bool(reply.get("empty")) or reply.get("error") == "no stamp found"

    def wait_clear(self, timeout=None):
        start = time.time()
        while not self.stop.is_set():
            if self.recal.is_set():  # Reset pressed: user says the trapdoor is empty
                self.recal.clear()
                self.calibrate()
                self.fingerprint()
                return True
            frame, presence, _ = self.measure(None)
            if presence < PRESENCE_FRAC:
                return True
            if timeout is not None and time.time() - start > timeout:
                return False
            time.sleep(0.15)
        return False

    # Sorting loop
    def run(self):
        try:
            if self.use_home:
                self.set("homing", "Homing carousel")
                self.hw.home()
            self.set("loading", "Loading classifier")
            ready = self.worker.wait_ready()
            with self.lock:
                self.state["threshold"] = ready.get("threshold", 0.70)
                self.state["classes"] = ready.get("classes", BIN_NAMES[:4])
            self.calibrate()
            self.fingerprint()
            self.loop()
        except Exception as error:
            self.set("error", f"Error: {error}")

    def waiting_message(self):
        if self.snapshot()["auto"]:
            return "Place a stamp on the trapdoor"
        return "Place a stamp on the trapdoor, then press Sort"

    def loop(self):
        prev, still_since = None, None
        self.set("waiting", self.waiting_message())
        while not self.stop.is_set():
            if self.recal.is_set():
                self.recal.clear()
                self.calibrate()
                self.fingerprint()
                prev, still_since = None, None
                self.set("waiting", self.waiting_message())
                continue
            frame, presence, motion = self.measure(prev)
            prev = frame
            auto = self.snapshot()["auto"]
            if self.sort_now.is_set():
                self.sort_now.clear()
                self.cycle(manual=True)
                prev, still_since = None, None
                continue
            if auto and presence >= PRESENCE_FRAC:
                if motion < STILL_MOTION:
                    still_since = still_since or time.time()
                    if time.time() - still_since >= STILL_TIME:
                        self.cycle(manual=False)
                        prev, still_since = None, None
                        continue
                    self.set("detected", "Something on the trapdoor, waiting for it to settle")
                else:
                    still_since = None
                    self.set("detected", "Something on the trapdoor, waiting for it to settle")
            else:
                still_since = None
                if presence < 0.02 and self.base is not None:
                    # Follow slow lighting changes without absorbing a stationary stamp.
                    self.base = 0.995 * self.base + 0.005 * frame
                self.set("waiting", self.waiting_message())
            time.sleep(0.1)

    def cycle(self, manual):
        start = time.time()
        if ROI is not None and not self.sim and self.base is not None:
            _, presence, _ = self.measure(None)  # last check: is anything really on the plate?
            if presence < PRESENCE_FRAC:
                self.set("nostamp", "Trapdoor looks empty. Nothing sorted")
                time.sleep(1.5)
                return
        self.set("photo", "Taking photo")
        photo = self.cam.capture()
        self.set("classifying", "Classifying")
        result = self.worker.classify(photo, ROI)
        if result.get("error") == "no stamp found" and not self.sim:
            # Retry once: a hand or a moving stamp can spoil the first photo.
            self.set("photo", "Stamp not found, taking another photo")
            time.sleep(1.0)
            photo = self.cam.capture()
            self.set("classifying", "Classifying")
            result = self.worker.classify(photo, ROI)
        bare_plate = (
            result.get("error") == "no stamp found"
            and result.get("roi_brightness", 999) < EMPTY_MAX_BRIGHT
        )
        # the classifier recognised the bare plate
        if result.get("empty") or bare_plate:
            # the plate moved a little; take a new empty photo
            self.calibrate()
            self.set("nostamp", "Trapdoor is empty. Nothing sorted")
            time.sleep(1.5)
            return
        if not result.get("ok"):
            self.set(
                "nostamp", f"No stamp found ({result.get('error', 'unknown')}). Check the trapdoor"
            )
            if self.sim:
                time.sleep(2)
                self.cam.dropped()
            elif ROI is not None:
                self.wait_clear()
            else:
                time.sleep(3)
            return
        best, confidence = result["best"], result["confidence"]
        target = best if result["accepted"] else BINS - 1
        crop_name = Path(result["crop"]).name
        self.next_id += 1
        entry_id = self.next_id
        entry = {
            "id": entry_id,
            "crop": crop_name,
            "bin": target,
            "name": BIN_NAMES[target],
            "best": best,
            "confidence": confidence,
            "scores": result["scores"],
            "accepted": result["accepted"],
            "classify_s": round(result["seconds"], 2),
        }
        with self.lock:
            self.state["last"] = entry
        self.set("moving", f"Stamp goes to bin {target} · {BIN_NAMES[target]}")
        self.hw.rotate_to(target)
        self.set("dropping", f"Dropping into bin {target} · {BIN_NAMES[target]}")
        self.hw.drop()
        with self.lock:
            self.state["counts"][target] += 1
            self.state["cycles"] += 1
            self.state["history"] = (
                [{k: entry[k] for k in ("id", "crop", "bin", "name", "confidence")}]
                + self.state["history"]
            )[:HISTORY]
            entry["cycle_s"] = round(time.time() - start, 1)
            cycles = self.state["cycles"]
        if ROI is not None or self.sim:
            self.set("checking", "Checking the trapdoor is empty")
            if not self.wait_clear(CLEAR_TIMEOUT):
                if not self.sim and self.plate_is_empty():
                    self.calibrate()  # empty, it just closed a bit differently
                else:
                    self.set("stuck", "Stamp did not fall. Clear the trapdoor")
                    self.wait_clear()
        if self.use_home and cycles % REHOME_EVERY == 0:
            self.set("homing", "Re-homing carousel")
            self.hw.home()

    def shutdown(self):
        self.stop.set()


# Web page
PAGE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Stamp Sorter</title>
    <style>
      :root {
        --bg: #0e0f11;
        --panel: #16181b;
        --line: #24272b;
        --text: #e8e9eb;
        --muted: #8b9096;
        --accent: #4c9aff;
        --ok: #3ecf8e;
        --warn: #f5a524;
        --bad: #f2555a;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font:
          15px/1.4 -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          Roboto,
          sans-serif;
      }

      .wrap {
        max-width: 960px;
        margin: 0 auto;
        padding: 20px 16px 28px;
      }

      header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
      }

      h1 {
        font-size: 15px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.02em;
        color: var(--muted);
      }

      .sp {
        flex: 1;
      }

      .switch {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
        font-size: 13px;
        cursor: pointer;
        user-select: none;
      }

      .switch i {
        width: 34px;
        height: 20px;
        border-radius: 10px;
        background: var(--line);
        position: relative;
        transition: 0.2s;
      }

      .switch i:after {
        content: "";
        position: absolute;
        top: 2px;
        left: 2px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #fff;
        transition: 0.2s;
      }

      .switch.on i {
        background: var(--accent);
      }

      .switch.on i:after {
        left: 16px;
      }

      .switch.dis {
        opacity: 0.4;
        cursor: default;
      }

      button {
        background: var(--text);
        color: var(--bg);
        border: 0;
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
      }

      button:disabled {
        opacity: 0.35;
        cursor: default;
      }

      button.ghost {
        background: transparent;
        color: var(--muted);
        border: 1px solid var(--line);
      }

      button.ghost:hover:not(:disabled) {
        color: var(--text);
      }

      button.ghost.off:hover {
        color: var(--bad);
        border-color: var(--bad);
      }

      .status {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 22px;
        font-weight: 600;
        margin: 0 0 18px;
      }

      .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--muted);
        flex: none;
      }

      .dot.busy {
        background: var(--accent);
        animation: p 1s infinite;
      }

      .dot.ready {
        background: var(--ok);
      }

      .dot.alert {
        background: var(--warn);
      }

      .dot.err {
        background: var(--bad);
      }

      @keyframes p {
        50% {
          opacity: 0.35;
        }
      }

      .main {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 16px;
      }

      .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 16px;
      }

      .photo {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 300px;
      }

      .photo img {
        max-width: 100%;
        max-height: 360px;
        border-radius: 6px;
      }

      .empty {
        color: var(--muted);
        font-size: 13px;
      }

      .verdict {
        font-size: 28px;
        font-weight: 700;
        margin: 2px 0 2px;
      }

      .sub {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 18px;
      }

      .bar {
        margin: 10px 0;
      }

      .bar .t {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 5px;
      }

      .bar .t b {
        font-weight: 600;
      }

      .track {
        position: relative;
        height: 8px;
        border-radius: 4px;
        background: var(--line);
      }

      .fill {
        height: 100%;
        border-radius: 4px;
        background: var(--muted);
        transition: width 0.3s;
      }

      .bar.win .fill {
        background: var(--accent);
      }

      .bar.win.rej .fill {
        background: var(--warn);
      }

      .thr {
        position: absolute;
        top: -4px;
        bottom: -4px;
        width: 2px;
        background: var(--text);
        opacity: 0.5;
      }

      .legend {
        font-size: 12px;
        color: var(--muted);
        margin-top: 12px;
      }

      .row {
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
        gap: 16px;
        margin-top: 16px;
      }

      .thumbs {
        display: flex;
        gap: 8px;
        overflow-x: auto;
      }

      .thumb {
        width: 72px;
        flex: none;
        text-align: center;
        font-size: 11px;
        color: var(--muted);
      }

      .thumb img {
        width: 72px;
        height: 72px;
        object-fit: contain;
        background: #000;
        border-radius: 6px;
        display: block;
        margin-bottom: 4px;
      }

      .bin {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: var(--muted);
        padding: 4px 0;
        border-bottom: 1px solid var(--line);
      }

      .bin:last-child {
        border: 0;
      }

      .bin b {
        color: var(--text);
      }

      .stale {
        opacity: 0.4;
        transition: opacity 0.3s;
      }

      .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 10px;
      }

      footer {
        margin-top: 16px;
        font-size: 11px;
        color: var(--muted);
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
      }

      @media (max-width: 700px) {
        .main,
        .row {
          grid-template-columns: 1fr;
        }

        .status {
          font-size: 18px;
        }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <h1>STAMP SORTER</h1>
        <span class="sp"></span>
        <div id="auto" class="switch"><i></i><span>Auto</span></div>
        <button id="sort">Sort now</button>
        <button id="reset" class="ghost">Reset</button>
        <button id="power" class="ghost off">Turn off</button>
      </header>

      <div class="status"><span id="dot" class="dot"></span><span id="msg">Connecting</span></div>

      <div class="main">
        <div class="card photo" id="photo"><span class="empty">No stamp yet</span></div>
        <div class="card" id="result">
          <div class="label">Classification</div>
          <div class="verdict" id="verdict">&nbsp;</div>
          <div class="sub" id="sub">&nbsp;</div>
          <div id="bars"></div>
          <div class="legend" id="legend"></div>
        </div>
      </div>

      <div class="row">
        <div class="card">
          <div class="label">Recent</div>
          <div class="thumbs" id="thumbs"><span class="empty">Nothing sorted yet</span></div>
        </div>
        <div class="card">
          <div class="label">Bins</div>
          <div id="counts"></div>
        </div>
      </div>

      <footer id="foot"></footer>
    </div>

    <script>
      const $ = (id) => document.getElementById(id);
      const BUSY_STATUSES = [
        "starting",
        "homing",
        "loading",
        "calibrating",
        "photo",
        "classifying",
        "moving",
        "dropping",
        "checking",
      ];
      const ALERT_STATUSES = ["detected", "nostamp", "stuck"];
      const BINS = ["nature", "people", "places", "graphic", "reject"];
      let lastStamp = null,
        historyKey = "",
        autoOn = false,
        autoAllowed = true,
        poweredOff = false;

      function capitalize(text) {
        return text.charAt(0).toUpperCase() + text.slice(1);
      }
      async function post(url, body) {
        await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body || {}),
        });
        tick();
      }
      $("sort").onclick = () => post("/api/sort");
      $("reset").onclick = () => {
        if (
          confirm(
            "Make sure the trapdoor is empty. Reset clears the counts and recent stamps and retakes the empty-trapdoor photo. Saved crops on the Pi are kept.",
          )
        )
          post("/api/reset");
      };
      $("power").onclick = () => {
        if (
          confirm(
            "Turn off the sorter program? You will need to start it again from the Pi terminal.",
          )
        ) {
          poweredOff = true;
          post("/api/shutdown");
        }
      };
      $("auto").onclick = () => {
        if (autoAllowed)
          post("/api/auto", {
            on: !autoOn,
          });
      };

      function render(state) {
        $("msg").textContent = state.message;
        $("dot").className =
          "dot " +
          (state.status === "error"
            ? "err"
            : ALERT_STATUSES.includes(state.status)
              ? "alert"
              : BUSY_STATUSES.includes(state.status)
                ? "busy"
                : "ready");
        autoOn = state.auto;
        autoAllowed = state.auto_allowed;
        $("auto").className = "switch" + (autoOn ? " on" : "") + (autoAllowed ? "" : " dis");
        $("auto").title = autoAllowed
          ? ""
          : "Auto needs the trapdoor area (ROI) set in sorter_app.py";
        $("sort").disabled = !(state.status === "waiting" || state.status === "detected");
        const stamp = state.last,
          threshold = state.threshold;
        $("reset").disabled = BUSY_STATUSES.includes(state.status);
        if (!stamp && lastStamp) {
          $("photo").innerHTML = '<span class="empty">No stamp yet</span>';
          $("verdict").innerHTML = "&nbsp;";
          $("sub").innerHTML = "&nbsp;";
          $("bars").innerHTML = "";
          $("legend").textContent = "";
          lastStamp = null;
        }
        if (stamp && (!lastStamp || lastStamp.id !== stamp.id)) {
          $("photo").innerHTML =
            '<img alt="stamp" src="/img/' +
            encodeURIComponent(stamp.crop) +
            "?v=" +
            stamp.id +
            '">';
          $("verdict").textContent = stamp.accepted
            ? "Bin " + stamp.bin + " · " + capitalize(stamp.name)
            : "Reject";
          $("sub").textContent = stamp.accepted
            ? Math.round(stamp.confidence * 100) +
              "% " +
              state.classes[stamp.best] +
              " · classified in " +
              stamp.classify_s +
              " s"
            : "Best guess " +
              state.classes[stamp.best] +
              " at " +
              (stamp.confidence * 100).toFixed(1) +
              "%, below " +
              Math.round(threshold * 100) +
              "% → bin 4";
          $("bars").innerHTML = state.classes
            .map((className, index) => {
              const score = stamp.scores[index];
              const isBest = index === stamp.best;
              return (
                '<div class="bar' +
                (isBest ? " win" : "") +
                (isBest && !stamp.accepted ? " rej" : "") +
                '"><div class="t"><span>' +
                capitalize(className) +
                "</span><b>" +
                (score * 100).toFixed(1) +
                '%</b></div><div class="track"><div class="fill" style="width:' +
                score * 100 +
                '%"></div><div class="thr" style="left:' +
                threshold * 100 +
                '%"></div></div></div>'
              );
            })
            .join("");
          $("legend").textContent =
            "Line = " +
            Math.round(threshold * 100) +
            "% threshold. Below it the stamp goes to the reject bin.";
          lastStamp = stamp;
        }
        const nextHistoryKey = state.history.map((entry) => entry.id).join(",");
        if (nextHistoryKey !== historyKey) {
          historyKey = nextHistoryKey;
          $("thumbs").innerHTML = state.history.length
            ? state.history
                .map(
                  (entry) =>
                    '<div class="thumb"><img alt="" src="/img/' +
                    encodeURIComponent(entry.crop) +
                    "?v=" +
                    entry.id +
                    '">Bin ' +
                    entry.bin +
                    "<br>" +
                    entry.name +
                    "</div>",
                )
                .join("")
            : '<span class="empty">Nothing sorted yet</span>';
        }
        $("counts").innerHTML = BINS.map(
          (binName, index) =>
            '<div class="bin"><span>Bin ' +
            index +
            " · " +
            capitalize(binName) +
            "</span><b>" +
            state.counts[index] +
            "</b></div>",
        ).join("");
        const isStale = ["photo", "classifying"].includes(state.status);
        $("photo").classList.toggle("stale", isStale);
        $("result").classList.toggle("stale", isStale);
        const metrics = state.metrics || {},
          parts = [state.cycles + " sorted"];
        if (stamp && stamp.cycle_s) parts.push("last cycle " + stamp.cycle_s + " s");
        if (metrics.presence !== undefined)
          parts.push("trapdoor " + metrics.presence + "% · motion " + metrics.motion);
        if (state.sim) parts.push("simulation");
        $("foot").innerHTML = parts.map((part) => "<span>" + part + "</span>").join("");
      }
      async function tick() {
        try {
          const response = await fetch("/api/state", {
            signal: AbortSignal.timeout(4000),
          });
          render(await response.json());
        } catch (error) {
          $("msg").textContent = poweredOff
            ? "Sorter is off. Start it again from the Pi terminal"
            : "Lost connection to the sorter";
          $("dot").className = poweredOff ? "dot" : "dot err";
        }
      }
      async function loop() {
        await tick();
        setTimeout(loop, 1000);
      }
      loop(); // one request at a time, slow links don't pile up
    </script>
  </body>
</html>"""


_jpeg_cache = {}


def small_jpeg(path, size=600):
    # Crop as a small JPEG for the web page (about 10x less data than the PNG).
    key = str(path)
    if key not in _jpeg_cache:
        try:
            import io
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((size, size))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=80)
            _jpeg_cache[key] = (buffer.getvalue(), "image/jpeg")
        except Exception:
            _jpeg_cache[key] = (path.read_bytes(), "image/png")
        if len(_jpeg_cache) > 50:
            _jpeg_cache.pop(next(iter(_jpeg_cache)))
    return _jpeg_cache[key]


def make_handler(sorter, crop_dir):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send(200, json.dumps(sorter.snapshot()).encode(), "application/json")
            elif self.path.startswith("/img/"):
                from urllib.parse import unquote

                name = os.path.basename(unquote(self.path[5:].split("?")[0]))
                path = crop_dir / name
                if name.endswith(".png") and path.is_file():
                    body, content_type = small_jpeg(path)
                    self.send(200, body, content_type, cache=True)
                else:
                    self.send(404, b"not found", "text/plain")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if self.path == "/api/sort":
                sorter.sort_now.set()
                self.send(200, b"{}", "application/json")
            elif self.path == "/api/auto":
                sorter.set_auto(body.get("on"))
                self.send(200, b"{}", "application/json")
            elif self.path == "/api/reset":
                sorter.reset()
                self.send(200, b"{}", "application/json")
            elif self.path == "/api/shutdown":
                sorter.set("off", "Turning off")
                self.send(200, b"{}", "application/json")
                sorter.quit.set()
            else:
                self.send(404, b"not found", "text/plain")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="no Pi hardware; photos from --photos")
    parser.add_argument("--photos", default="sim_photos", help="folder of photos for --sim")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="skip homing (home sensor not mounted); the carousel's current spot is bin 0",
    )
    parser.add_argument(
        "--worker-python",
        default=None,
        help="Python for the classifier (default: ~/clip-env/bin/python, or this Python with --sim)",
    )
    parser.add_argument("--worker", default=str(HERE / "classifier_worker.py"))
    args = parser.parse_args()

    python = args.worker_python or (sys.executable if args.sim else CLIP_PYTHON)
    python = python if os.path.exists(python) else (shutil.which(python) or python)
    if not os.path.exists(python):
        raise SystemExit(f"Classifier Python not found: {python}")

    worker = Worker(python, args.worker)  # starts loading CLIP right away
    hardware = SimHardware() if args.sim else Hardware()
    camera = SimCamera(args.photos) if args.sim else Camera()
    if args.sim:
        hardware.on_drop = camera.dropped

    sorter = Sorter(hardware, camera, worker, args.sim, use_home=not args.no_home)
    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port), make_handler(sorter, Path(args.worker).resolve().parent / "crops")
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"Sorter page: http://thinh.local:{args.port}  (or http://<Pi IP>:{args.port}; "
        f"http://localhost:{args.port} on this computer)"
    )
    print("Press Ctrl+C to stop.")
    machine = threading.Thread(target=sorter.run, daemon=True)
    machine.start()
    try:
        while machine.is_alive() and not sorter.quit.is_set():
            machine.join(0.5)
        if not sorter.quit.is_set():
            print("Sorter stopped:", sorter.snapshot()["message"])
        while not sorter.quit.is_set():  # keep the page up to show the error
            time.sleep(1)
        print("\nTurned off from the web page. Stopping...")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sorter.shutdown()
        machine.join(15)  # let a stamp in progress finish
        for part in (hardware, camera, worker):
            try:
                part.close()
            except Exception as error:
                print("cleanup:", error)
        server.shutdown()
        print("Trapdoor closed, driver disabled.")


if __name__ == "__main__":
    main()
