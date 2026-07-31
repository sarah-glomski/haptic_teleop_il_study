#!/usr/bin/env python3
"""
Live rollout monitor — wrist camera + piezense force, side by side.

Runs as its OWN process with its OWN window, launched/killed by pressing L in
inference.py's control window (or run directly). Separate process on purpose:

  • The control window must never stall. A camera decode or a plot redraw
    sharing that pygame loop makes a slow frame a slow ARM response, and a
    rollout cannot afford that.
  • It subscribes to the topics itself rather than being fed by inference, so
    it works during teleop, during data collection, or standalone.

Usage:
    python live_viewer.py                     # 1180x620 default
    python live_viewer.py --seconds 15        # shorter force history
"""

import argparse
import os
import time

import cv2
import numpy as np
import pygame
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from piezense_interfaces.msg import PiezenseSystemArray

PIEZENSE_TOPIC          = "piezense/data"
PIEZENSE_SYSTEM_ID      = 0
PIEZENSE_INPUT_CHAN_IDS = [2, 3]
WRIST_TOPIC             = "/dji_wrist/dji_wrist/color/image_raw"

# Categorical slots 1 and 2 of the reference palette, stepped for a dark
# surface, in fixed order. Both channels share ONE Pa axis — same unit, so two
# scales would misrepresent their relative magnitude.
COLORS   = ((0x39, 0x87, 0xe5), (0xd9, 0x59, 0x26))
SURFACE  = (0x1a, 0x1a, 0x19)
PANEL    = (0x24, 0x24, 0x22)
GRID     = (0x3a, 0x3a, 0x38)
INK      = (0xff, 0xff, 0xff)
INK_MUTE = (0xc3, 0xc2, 0xb7)
INK_DIM  = (0x8a, 0x8a, 0x82)
OK_C     = (0x1b, 0xaf, 0x7a)
BAD_C    = (0xe3, 0x49, 0x48)

MIN_SPAN_PA = 2000.0     # never zoom tighter, or idle noise fills the plot
STALE_S     = 3.0
REC_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "live_recordings")


class ScreenRecorder:
    """MP4 of the live pane, toggled by V. First press starts, second saves.

    Frames go to a hidden `.part` file and are only renamed into place on the
    SECOND press. Any other exit — window closed, Q/L/Esc, Ctrl-C — deletes it,
    so an unfinished recording is never left looking like a finished one.
    """

    def __init__(self, out_dir: str, fps: int):
        self.out_dir = out_dir
        self.fps     = fps
        self._writer = None
        self._tmp    = None
        self._final  = None
        self._size   = None      # locked at start; the window may be resized
        self._t0     = None
        self._frames = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def start(self, size):
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._final = os.path.join(self.out_dir, f"live_{stamp}.mp4")
        self._tmp   = os.path.join(self.out_dir, f".live_{stamp}.part.mp4")
        self._size  = size
        writer = cv2.VideoWriter(self._tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, size)
        if not writer.isOpened():
            self._writer = self._tmp = self._final = self._size = None
            return None
        self._writer, self._t0, self._frames = writer, time.monotonic(), 0
        return self._final

    def capture(self, screen):
        if self._writer is None:
            return
        # (W,H,3) RGB -> (H,W,3) BGR, resized if the window changed mid-record.
        arr = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
        if (arr.shape[1], arr.shape[0]) != self._size:
            arr = cv2.resize(arr, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        self._frames += 1

    def stop(self, save: bool):
        """Returns (path, frames, seconds) when saved, else None."""
        if self._writer is None:
            return None
        secs, frames = self.elapsed(), self._frames
        self._writer.release()
        self._writer = None
        result = None
        try:
            if save and frames > 0:
                os.replace(self._tmp, self._final)
                result = (self._final, frames, secs)
            elif self._tmp and os.path.exists(self._tmp):
                os.unlink(self._tmp)
        except OSError:
            pass
        self._tmp = self._final = self._size = self._t0 = None
        self._frames = 0
        return result


class LiveViewer(Node):
    def __init__(self, seconds: float, hz: int):
        super().__init__("live_viewer")
        self.maxlen  = int(seconds * hz)
        self.seconds = seconds
        self.hist    = []
        self.latest  = None
        self.pz_rx   = None

        self._bridge = CvBridge()
        self.frame   = None          # HWC RGB uint8
        self.cam_rx  = None

        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PiezenseSystemArray, PIEZENSE_TOPIC, self._pz_cb, 10)
        self.create_subscription(Image, WRIST_TOPIC, self._cam_cb, sensor_qos)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _pz_cb(self, msg):
        for sys_msg in msg.system:
            if sys_msg.system_id == PIEZENSE_SYSTEM_ID:
                r = list(sys_msg.pressure_pa)
                self.latest = tuple(float(r[c]) if c < len(r) else 0.0
                                    for c in PIEZENSE_INPUT_CHAN_IDS)
                break
        self.pz_rx = self._now()

    def _cam_cb(self, msg):
        try:
            self.frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.cam_rx = self._now()
        except Exception:
            pass

    def sample(self):
        """Sampled at the DRAW rate so the x axis is uniform in time, not in
        message arrivals (the sensor's rate is not guaranteed steady)."""
        if self.latest is not None:
            self.hist.append(self.latest)
            if len(self.hist) > self.maxlen:
                del self.hist[:len(self.hist) - self.maxlen]

    def pz_live(self):
        return self.pz_rx is not None and (self._now() - self.pz_rx) < STALE_S

    def cam_live(self):
        return self.cam_rx is not None and (self._now() - self.cam_rx) < STALE_S


def _chip(screen, font, label, live, x, y):
    """Status pill. Text carries the state, colour only reinforces it."""
    text = f"{label} {'LIVE' if live else 'NO DATA'}"
    surf = font.render(text, True, OK_C if live else BAD_C)
    screen.blit(surf, (x - surf.get_width(), y))
    return surf.get_width()


def _draw_camera(screen, fonts, node, rect):
    font, font_sm, _ = fonts
    x, y, side = rect
    pygame.draw.rect(screen, PANEL, (x, y, side, side))
    screen.blit(font_sm.render("Wrist camera", True, INK_MUTE), (x, y - 22))

    frame = node.frame
    if frame is None:
        msg = font_sm.render("waiting for frames …", True, INK_DIM)
        screen.blit(msg, (x + side // 2 - msg.get_width() // 2, y + side // 2))
        return
    # (H,W,3) -> pygame wants (W,H,3)
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(pygame.transform.smoothscale(surf, (side, side)), (x, y))
    if not node.cam_live():
        # Say it in words — a frozen image looks identical to a live still one.
        warn = font_sm.render("STALE", True, BAD_C)
        screen.blit(warn, (x + 8, y + 8))


def _draw_plot(screen, fonts, node, left, right, top, bot):
    font, font_sm, _ = fonts
    pw, ph = right - left, bot - top
    pygame.draw.rect(screen, PANEL, (left, top, pw, ph))
    screen.blit(font_sm.render("Piezense force (Pa)", True, INK_MUTE), (left, top - 22))

    hist = node.hist
    if len(hist) < 2:
        msg = font_sm.render("waiting for piezense/data …", True, INK_DIM)
        screen.blit(msg, (left + pw // 2 - msg.get_width() // 2, top + ph // 2))
        return

    lo = min(min(p) for p in hist)
    hi = max(max(p) for p in hist)
    mid, span = (lo + hi) / 2.0, max(hi - lo, MIN_SPAN_PA)
    lo, hi = mid - span / 2.0, mid + span / 2.0

    for k in range(5):
        frac = k / 4.0
        y = int(bot - frac * ph)
        pygame.draw.line(screen, GRID, (left, y), (right, y), 1)
        lab = font_sm.render(f"{lo + frac * span:,.0f}", True, INK_DIM)
        screen.blit(lab, (left - 8 - lab.get_width(), y - 8))
        x = int(left + frac * pw)
        pygame.draw.line(screen, GRID, (x, top), (x, bot), 1)
        secs = node.seconds * (1.0 - frac)
        xlab = font_sm.render("now" if secs < 0.1 else f"-{secs:.0f}s", True, INK_DIM)
        screen.blit(xlab, (x - xlab.get_width() // 2, bot + 8))

    step = pw / max(1, node.maxlen - 1)
    x0   = right - (len(hist) - 1) * step
    for ch in (0, 1):
        pts = [(x0 + i * step, bot - (max(lo, min(hi, p[ch])) - lo) / span * ph)
               for i, p in enumerate(hist)]
        pygame.draw.lines(screen, COLORS[ch], False, pts, 2)

    # Legend with live values. Text wears ink tokens; the swatch carries
    # identity, so the series are never distinguished by colour alone.
    lx = left
    for ch, label in enumerate(("channel 2", "channel 3")):
        pygame.draw.rect(screen, COLORS[ch], (lx, bot + 34, 12, 12))
        surf = font.render(f"{label}   {hist[-1][ch]:,.0f} Pa", True, INK_MUTE)
        screen.blit(surf, (lx + 20, bot + 32))
        lx += 20 + surf.get_width() + 34


def draw_rec_indicator(screen, font, recorder, w):
    """Blinking red dot + elapsed time. Drawn AFTER the frame is captured, so
    the saved MP4 stays clean and the marker is screen-only."""
    secs = recorder.elapsed()
    if int(secs * 2) % 2 == 0:                      # ~1 Hz blink
        pygame.draw.circle(screen, BAD_C, (34, 60), 8)
    label = font.render(f"REC  {int(secs // 60):01d}:{int(secs % 60):02d}",
                        True, BAD_C)
    screen.blit(label, (50, 52))


def draw(screen, fonts, node, w, h):
    font, font_sm, font_big = fonts
    screen.fill(SURFACE)
    screen.blit(font_big.render("Live", True, INK), (24, 18))

    used = _chip(screen, font, "force", node.pz_live(), w - 24, 26)
    _chip(screen, font, "cam", node.cam_live(), w - 24 - used - 22, 26)

    pad, head, foot = 24, 74, 78
    body_h   = h - head - foot
    cam_side = max(120, min(body_h, int(w * 0.38)))
    _draw_camera(screen, fonts, node, (pad, head, cam_side))

    plot_left = pad + cam_side + 96      # room for the y labels, drawn outside
    if w - pad - plot_left > 160:
        _draw_plot(screen, fonts, node, plot_left, w - pad, head, head + body_h)


def main():
    ap = argparse.ArgumentParser(description="Live wrist camera + piezense force")
    ap.add_argument("--seconds", type=float, default=30.0, help="visible history")
    ap.add_argument("--hz", type=int, default=30, help="draw/sample rate")
    ap.add_argument("--width", type=int, default=1180)
    ap.add_argument("--height", type=int, default=620)
    ap.add_argument("--record-dir", type=str, default=REC_DIR_DEFAULT,
                    help="where V saves MP4s (default: testing/live_recordings)")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = LiveViewer(args.seconds, args.hz)
    recorder = ScreenRecorder(args.record_dir, args.hz)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    pygame.display.set_caption("Live — wrist cam + piezense")
    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont("monospace", 17),
             pygame.font.SysFont("monospace", 14),
             pygame.font.SysFont("monospace", 26, bold=True))

    w, h = args.width, args.height
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.VIDEORESIZE:
                    w, h = event.w, event.h
                    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                if event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_q, pygame.K_l, pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    if recorder.active:
                        done = recorder.stop(save=True)
                        print(f"Saved {done[0]}  ({done[1]} frames, "
                              f"{done[2]:.1f}s)" if done else
                              "Recording had no frames — nothing saved",
                              flush=True)
                    else:
                        path = recorder.start((w, h))
                        print(f"Recording -> {path}  (V again to save)" if path
                              else "Could not open the MP4 writer", flush=True)
            rclpy.spin_once(node, timeout_sec=0.0)
            node.sample()
            draw(screen, fonts, node, w, h)
            if recorder.active:
                # Capture BEFORE the REC marker so the MP4 has no overlay.
                recorder.capture(screen)
                draw_rec_indicator(screen, fonts[0], recorder, w)
            pygame.display.flip()
            clock.tick(args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        # Any exit that is not a second V press DISCARDS the recording.
        dropped = recorder.active
        recorder.stop(save=False)
        if dropped:
            print("Recording discarded (stopped before the second V).", flush=True)
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()
