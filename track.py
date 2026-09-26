"""Track the nearest detected person with a MIPC PTZ camera.

YOLO runs on a separate, fixed overview camera that faces the same general
direction as the MIPC pan/tilt camera (not an overhead/map view). The MIPC
camera's boresight sits at a known pixel column (--cam-x) within that
overview frame; since both cameras face the same way, only the target's
*horizontal* pixel offset from that column maps to a pan angle (vertical
offset is depth, not direction, and is ignored for panning). That offset
is converted to degrees via perspective projection using the overview
camera's horizontal field of view (--hfov-deg).

control_ptz() moves are relative raw-unit offsets with no position
feedback from the device, so the current pan angle is tracked in software
(as plain degrees) and converted to raw units via --pan-range-units. To
get a known starting reference, the script homes the camera at startup by
driving it past its leftmost hard stop, then swings back to
straight-forward. The hard stop's angle (--home-pan-deg) is a hardware
fact that must be calibrated separately from --backward-margin-deg, which
is just the desired soft safety margin used during tracking.
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Any

from ultralytics import YOLO

from mipc_camera_client import MipcCameraClient

LOGGER = logging.getLogger("track")

PERSON_CLASS_ID = 0
RECENTER_EPSILON_DEG = 0.5


def normalize_deg(angle: float) -> float:
    """Wrap an angle to (-180, 180]."""
    return ((angle + 180.0) % 360.0) - 180.0


@dataclass
class Config:
    source: str
    model: str
    cam_x: float
    cam_y: float
    hfov_deg: float
    forward_deg: float
    backward_margin_deg: float
    tick_hz: float
    recenter_after_s: float
    max_step_deg: float
    conf: float
    speed_x: int
    speed_y: int
    home_seconds: float
    center_seconds: float
    home_pan_deg: float
    pan_range_units: float
    host: str
    user: str
    password: str

    @property
    def min_pan_deg(self) -> float:
        return -(180.0 - self.backward_margin_deg)

    @property
    def max_pan_deg(self) -> float:
        return 180.0 - self.backward_margin_deg

    @property
    def units_per_deg(self) -> float:
        """control_ptz's tilt_x is in raw camera units, not real degrees.
        pan_range_units is the measured full sweep (min_pan_deg to
        max_pan_deg) in those units, so this converts a delta in our
        internal degree model to the raw units to actually send."""
        return self.pan_range_units / (self.max_pan_deg - self.min_pan_deg)


def _env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value is not None else None


def parse_args() -> tuple[Config, Any]:
    parser = argparse.ArgumentParser(
        description="Pan a MIPC PTZ camera to track the person closest to it, "
        "using YOLO detections from a separate overview camera."
    )
    parser.add_argument(
        "--source",
        default=os.getenv("YOLO_SOURCE"),
        help="video source for YOLO (webcam index, RTSP/RTMP URL, file), or YOLO_SOURCE env var",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("YOLO_MODEL", "yolo26n.pt"),
        help="ultralytics model path/name (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--cam-x",
        type=float,
        default=_env_float("CAM_X"),
        help="x pixel position of the mipc camera within the YOLO frame, or CAM_X env var",
    )
    parser.add_argument(
        "--cam-y",
        type=float,
        default=_env_float("CAM_Y"),
        help="y pixel position of the mipc camera within the YOLO frame (used only to pick "
        "the closest detected person, not for pan angle), or CAM_Y env var",
    )
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=_env_float("YOLO_HFOV_DEG"),
        help="horizontal field of view, in degrees, of the overview camera feeding YOLO -- "
        "used to convert a target's horizontal pixel offset from --cam-x into a pan angle, "
        "or YOLO_HFOV_DEG env var",
    )
    parser.add_argument(
        "--forward-deg",
        type=float,
        default=float(os.getenv("CAM_FORWARD_DEG", "0")),
        help="fine trim (degrees) added to the computed pan angle, in case --cam-x isn't "
        "exactly on the camera's true pan=0 column (default: 0, no trim)",
    )
    parser.add_argument(
        "--backward-margin-deg",
        type=float,
        default=15.0,
        help="degrees on either side of straight-backward the camera may not pan into (default: 15)",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=5.0,
        help="how many times per second to issue pan commands (default: 5)",
    )
    parser.add_argument(
        "--recenter-after",
        type=float,
        default=2.5,
        help="seconds with no person detected before recentering to forward (default: 2.5)",
    )
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=6.0,
        help="max degrees to pan per tick, for smooth tracking (default: 6)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="YOLO detection confidence threshold (default: 0.4)",
    )
    parser.add_argument("--speed-x", type=int, default=80, help="mipc pan motor speed (default: 80)")
    parser.add_argument("--speed-y", type=int, default=80, help="mipc tilt motor speed (default: 80)")
    parser.add_argument(
        "--home-seconds",
        type=float,
        default=6.0,
        help="seconds to wait for the startup homing sweep to reach the hard stop "
        "(default: 6; tune to how long a full pan sweep takes on your hardware)",
    )
    parser.add_argument(
        "--center-seconds",
        type=float,
        default=6.0,
        help="seconds to wait for the post-homing swing to straight-forward to complete "
        "(default: 6; increase if the camera still hasn't reached forward when tracking "
        "starts, decrease once you've confirmed it reliably finishes early)",
    )
    parser.add_argument(
        "--home-pan-deg",
        type=float,
        default=None,
        help="the actual pan angle (degrees from straight-forward) that homing reaches at "
        "the leftmost hard stop -- this is a hardware fact, independent of "
        "--backward-margin-deg (a desired safety margin), and controls how far the "
        "post-homing swing-to-forward travels. Defaults to -(180 - backward-margin-deg); "
        "if centering isn't reaching forward, increase this (more negative) to command a "
        "larger swing",
    )
    parser.add_argument(
        "--pan-range-units",
        type=float,
        default=475.0,
        help="measured full range of motion for tilt_x, in raw camera units, spanning "
        "min-pan-deg to max-pan-deg (default: 475). control_ptz's tilt_x is not real "
        "degrees; this is used to convert commanded degree deltas to the raw units the "
        "camera expects",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CAMERA_HOST"),
        help="mipc camera address, or CAMERA_HOST env var",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("CAMERA_USER"),
        help="mipc camera username, or CAMERA_USER env var",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="silence info logging")
    args = parser.parse_args()

    missing = [
        name
        for name, value in [
            ("--source/YOLO_SOURCE", args.source),
            ("--cam-x/CAM_X", args.cam_x),
            ("--cam-y/CAM_Y", args.cam_y),
            ("--hfov-deg/YOLO_HFOV_DEG", args.hfov_deg),
            ("--host/CAMERA_HOST", args.host),
            ("--user/CAMERA_USER", args.user),
        ]
        if value is None
    ]
    if missing:
        parser.error(f"missing required options: {', '.join(missing)}")
    password = os.environ.get("CAMERA_PASSWORD")
    if not password:
        parser.error("environment variable CAMERA_PASSWORD must be set")

    return Config(
        source=args.source,
        model=args.model,
        cam_x=args.cam_x,
        cam_y=args.cam_y,
        hfov_deg=args.hfov_deg,
        forward_deg=args.forward_deg,
        backward_margin_deg=args.backward_margin_deg,
        tick_hz=args.tick_hz,
        recenter_after_s=args.recenter_after,
        max_step_deg=args.max_step_deg,
        conf=args.conf,
        speed_x=args.speed_x,
        speed_y=args.speed_y,
        home_seconds=args.home_seconds,
        center_seconds=args.center_seconds,
        home_pan_deg=(
            args.home_pan_deg if args.home_pan_deg is not None else -(180.0 - args.backward_margin_deg)
        ),
        pan_range_units=args.pan_range_units,
        host=args.host,
        user=args.user,
        password=password,
    ), args.quiet


def pick_closest_person(result, cam_x: float, cam_y: float):
    """Returns the (x, y) ground position of the detected person nearest to
    (cam_x, cam_y), using bbox bottom-center as a proxy for foot position."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_dist = math.inf
    for x1, y1, x2, y2 in boxes.xyxy.tolist():
        px = (x1 + x2) / 2.0
        py = y2
        dist = math.hypot(px - cam_x, py - cam_y)
        if dist < best_dist:
            best_dist = dist
            best = (px, py)
    return best


def bearing_to_target(cam_x: float, target_x: float, frame_width: float, hfov_deg: float, forward_deg: float) -> float:
    """Pan angle from the camera's boresight to the target, in degrees,
    where positive is clockwise (matching control_ptz's tilt_x sign).

    Both cameras face the same general direction, so only the target's
    horizontal pixel offset from cam_x matters -- converted to an angle
    via perspective projection using the overview camera's HFOV."""
    focal_px = (frame_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    angle = math.degrees(math.atan2(target_x - cam_x, focal_px))
    return normalize_deg(angle - forward_deg)


def step_toward(current: float, target: float, max_step: float) -> float:
    """Step from current toward target, both assumed to already lie within
    [min_pan_deg, max_pan_deg]. Plain subtraction, not normalize_deg -- our
    pan range is a bounded arc, not a full circle, so there's no "shorter
    way around" through the blocked backward zone to wrap toward."""
    delta = target - current
    return max(-max_step, min(max_step, delta))


def home_camera(camera: MipcCameraClient, cfg: Config) -> float:
    """Drive the camera past its leftmost hard stop (and tilt to its top
    hard stop) so we have a known reference position to track pan angle
    from, and return that angle (cfg.home_pan_deg, a calibrated hardware
    fact -- see --home-pan-deg).
    """
    LOGGER.info(f"Homing camera to leftmost/top hard stop (waiting {cfg.home_seconds:.1f}s)")
    camera.control_ptz(tilt_x=-360, tilt_y=-200, speed_x=cfg.speed_x, speed_y=cfg.speed_y)
    time.sleep(cfg.home_seconds)
    return cfg.home_pan_deg


def center_camera(camera: MipcCameraClient, current_pan_deg: float, cfg: Config) -> float:
    """Swing the camera from its current (known) pan angle to straight
    forward (0 deg), drop tilt back down from its homed top position, and
    block until the move should be complete."""
    delta = 0.0 - current_pan_deg
    LOGGER.info(f"Centering camera to forward ({delta:+.1f} deg, waiting {cfg.center_seconds:.1f}s)")
    camera.control_ptz(tilt_x=delta * cfg.units_per_deg, tilt_y=80, speed_x=cfg.speed_x, speed_y=cfg.speed_y)
    time.sleep(cfg.center_seconds)
    return 0.0


def run(cfg: Config) -> None:
    LOGGER.info(f"Logging into mipc camera at {cfg.host}")
    camera = MipcCameraClient(cfg.host)
    camera.login(cfg.user, cfg.password)

    model = YOLO(cfg.model)

    current_pan_deg = home_camera(camera, cfg)
    current_pan_deg = center_camera(camera, current_pan_deg, cfg)
    last_seen: Optional[float] = None
    last_tick = 0.0
    tick_interval = 1.0 / cfg.tick_hz
    recentered = True

    LOGGER.info(f"Starting detection stream from {cfg.source}")
    results = model.predict(
        source=cfg.source,
        stream=True,
        classes=[PERSON_CLASS_ID],
        conf=cfg.conf,
        verbose=False,
        show=True,
    )

    for result in results:
        now = time.monotonic()
        if now - last_tick < tick_interval:
            continue
        last_tick = now

        target_point = pick_closest_person(result, cfg.cam_x, cfg.cam_y)

        if target_point is not None:
            last_seen = now
            recentered = False
            frame_width = result.orig_shape[1]
            target_deg = bearing_to_target(cfg.cam_x, target_point[0], frame_width, cfg.hfov_deg, cfg.forward_deg)
            target_deg = max(cfg.min_pan_deg, min(cfg.max_pan_deg, target_deg))
        elif not recentered and last_seen is not None and now - last_seen >= cfg.recenter_after_s:
            target_deg = 0.0
        else:
            continue

        delta = step_toward(current_pan_deg, target_deg, cfg.max_step_deg)
        if abs(delta) < RECENTER_EPSILON_DEG:
            if target_deg == 0.0:
                recentered = True
            continue

        LOGGER.info(f"panning {delta:+.1f} deg (current={current_pan_deg:.1f} target={target_deg:.1f})")
        camera.control_ptz(tilt_x=delta * cfg.units_per_deg, tilt_y=0, speed_x=cfg.speed_x)
        current_pan_deg = normalize_deg(current_pan_deg + delta)
        if current_pan_deg == 0.0:
            recentered = True


def main() -> None:
    cfg, quiet = parse_args()
    level = logging.CRITICAL if quiet else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=level,
    )
    run(cfg)


if __name__ == "__main__":
    main()
