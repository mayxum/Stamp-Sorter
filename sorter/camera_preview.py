# Live camera view for mounting the camera and measuring the trapdoor box (ROI).
#
# On the Pi (system Python; stop sorter_app.py first, only one program can use the camera):
#     python3 ~/sorter/camera_preview.py
# then open http://thinh.local:8001 on the Mac (or http://<Pi IP>:8001 on the hotspot).
#
# The page shows the live feed with a center crosshair and thirds grid. Drag a box
# around the trapdoor and it prints the ROI in full-resolution pixels, ready to paste
# into sorter_app.py (ROI = (x, y, w, h)).
#
# Options:
#     --auto-exposure   let the camera choose exposure (default: the sorter's 1/100 s, EV -1)
#     --small           640 x 360 at 5 frames/s, for a weak hotspot
#     --port 8001

import argparse
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FULL_SIZE = (4608, 2592)  # same as sorter_app.py; ROI is reported in these pixels
VIEW_SIZE = (1280, 720)  # streamed size (same field of view, scaled down)
EXPOSURE_US = 10000
EV = -1.0

PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Camera View</title>
    <style>
      :root {
        --bg: #0e0f11;
        --text: #e8e9eb;
        --muted: #8b9096;
        --accent: #4c9aff;
        --line: #24272b;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font:
          14px/1.4 -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          Roboto,
          sans-serif;
      }
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 16px;
      }
      header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }
      h1 {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: var(--muted);
        margin: 0;
      }
      .sp {
        flex: 1;
      }
      label {
        color: var(--muted);
        font-size: 13px;
        display: flex;
        gap: 6px;
        align-items: center;
        cursor: pointer;
      }
      .view {
        position: relative;
        width: 100%;
        aspect-ratio: 16/9;
        background: #000;
        border-radius: 10px;
        overflow: hidden;
        cursor: crosshair;
        user-select: none;
      }
      .view img,
      .view canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }
      .roi {
        margin-top: 12px;
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
        color: var(--muted);
      }
      code {
        background: #16181b;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 6px 10px;
        color: var(--text);
        font-size: 14px;
      }
      button {
        background: var(--text);
        color: var(--bg);
        border: 0;
        border-radius: 8px;
        padding: 7px 12px;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
      }
      .hint {
        color: var(--muted);
        font-size: 12px;
        margin-top: 8px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <h1>CAMERA VIEW</h1>
        <span class="sp"></span><label><input type="checkbox" id="grid" checked /> Grid</label>
      </header>
      <div class="view" id="view">
        <img id="img" src="/stream.mjpg" alt="live camera" /><canvas id="cv"></canvas>
      </div>
      <div class="roi">
        <span>Trapdoor box:</span><code id="roi">drag a box on the image</code
        ><button id="copy">Copy</button>
      </div>
      <div class="hint">
        Drag around the trapdoor plate (a little margin is fine). Paste the line into sorter_app.py.
      </div>
    </div>
    <script>
      const FRAME_WIDTH = __FW__,
        FRAME_HEIGHT = __FH__;
      const view = document.getElementById("view"),
        canvas = document.getElementById("cv"),
        context = canvas.getContext("2d");
      let box = null,
        dragStart = null;
      function resizeCanvas() {
        const bounds = view.getBoundingClientRect();
        canvas.width = bounds.width * devicePixelRatio;
        canvas.height = bounds.height * devicePixelRatio;
        draw();
      }
      function draw() {
        const width = canvas.width,
          height = canvas.height;
        context.clearRect(0, 0, width, height);
        if (document.getElementById("grid").checked) {
          context.strokeStyle = "rgba(255,255,255,.25)";
          context.lineWidth = devicePixelRatio;
          for (const fraction of [1 / 3, 2 / 3]) {
            context.beginPath();
            context.moveTo(width * fraction, 0);
            context.lineTo(width * fraction, height);
            context.moveTo(0, height * fraction);
            context.lineTo(width, height * fraction);
            context.stroke();
          }
          context.strokeStyle = "rgba(255,255,255,.7)";
          const crosshairSize = 16 * devicePixelRatio;
          context.beginPath();
          context.moveTo(width / 2 - crosshairSize, height / 2);
          context.lineTo(width / 2 + crosshairSize, height / 2);
          context.moveTo(width / 2, height / 2 - crosshairSize);
          context.lineTo(width / 2, height / 2 + crosshairSize);
          context.stroke();
        }
        if (box) {
          context.strokeStyle = "#4c9aff";
          context.lineWidth = 2 * devicePixelRatio;
          context.strokeRect(box.x * width, box.y * height, box.w * width, box.h * height);
        }
      }
      function relativePosition(event) {
        const bounds = view.getBoundingClientRect();
        return {
          x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
          y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
        };
      }
      view.addEventListener("pointerdown", (event) => {
        dragStart = relativePosition(event);
        view.setPointerCapture(event.pointerId);
      });
      view.addEventListener("pointermove", (event) => {
        if (!dragStart) return;
        const position = relativePosition(event);
        box = {
          x: Math.min(dragStart.x, position.x),
          y: Math.min(dragStart.y, position.y),
          w: Math.abs(position.x - dragStart.x),
          h: Math.abs(position.y - dragStart.y),
        };
        draw();
      });
      view.addEventListener("pointerup", () => {
        dragStart = null;
        if (box && box.w > 0.01 && box.h > 0.01) {
          const bounds = [
            box.x * FRAME_WIDTH,
            box.y * FRAME_HEIGHT,
            box.w * FRAME_WIDTH,
            box.h * FRAME_HEIGHT,
          ].map(Math.round);
          document.getElementById("roi").textContent = "ROI = (" + bounds.join(", ") + ")";
        }
      });
      document.getElementById("grid").onchange = draw;
      document.getElementById("copy").onclick = () =>
        navigator.clipboard &&
        navigator.clipboard.writeText(document.getElementById("roi").textContent);
      window.addEventListener("resize", resizeCanvas);
      resizeCanvas();
    </script>
  </body>
</html>"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


def make_handler(output):
    page = PAGE.replace("__FW__", str(FULL_SIZE[0])).replace("__FH__", str(FULL_SIZE[1])).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame
                        self.wfile.write(
                            b"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def start_camera(output, auto_exposure, small=False):
    from libcamera import controls
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput

    camera = Picamera2()
    try:  # full-field binned sensor mode, so the view matches the sorter's 4608 x 2592 frame
        config = camera.create_video_configuration(
            main={"size": (640, 360) if small else VIEW_SIZE},
            sensor={"output_size": (FULL_SIZE[0] // 2, FULL_SIZE[1] // 2)},
        )
    except TypeError:
        config = camera.create_video_configuration(
            main={"size": (640, 360) if small else VIEW_SIZE}
        )
    camera.configure(config)
    settings = {"AfMode": controls.AfModeEnum.Continuous, "AfRange": controls.AfRangeEnum.Full}
    if not auto_exposure:
        settings.update({"ExposureTime": EXPOSURE_US, "ExposureValue": EV})
    if small:
        settings["FrameDurationLimits"] = (200000, 200000)  # 5 frames per second
    camera.set_controls(settings)
    camera.start_recording(MJPEGEncoder(), FileOutput(output))
    return camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument(
        "--small", action="store_true", help="640 x 360 at 5 fps, for a weak hotspot"
    )
    args = parser.parse_args()

    output = StreamingOutput()
    camera = start_camera(output, args.auto_exposure, args.small)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(output))
    server.daemon_threads = True
    print(f"Camera view: http://thinh.local:{args.port}  (or http://<Pi IP>:{args.port})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop_recording()
        print("\nCamera stopped.")


if __name__ == "__main__":
    main()
