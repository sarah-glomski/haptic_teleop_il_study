#!/usr/bin/env python3
"""
Release a leaked Bluetooth link to the piezense before launching its driver.

The piezense connects over BLE. Its driver (piezense_ros/ar_teleop.py) wraps
the main loop in `except Exception`, and KeyboardInterrupt is a BaseException —
so Ctrl-C skips its disconnect() and BlueZ keeps the link open after the
process is gone. A connected BLE peripheral stops advertising, so the NEXT
launch scans, finds nothing, and spins on "[example] connecting..." forever
with no indication why.

Every Ctrl-C leaks it. It works after a reboot or a sensor power-cycle and
fails on the next relaunch, which makes it look intermittent and hardware-ish.
Observed twice: 3761 retries on 2026-08-13 during data collection, then again
on 2026-08-14 when inference inherited the link the collector had left behind.

Cleaned up here rather than upstream: piezense_ros is a shared package, and
this is our launchers' mess to clean. Lives in its own module because all
three launchers start the driver and all three need it — data collection had
the fix first and the other two kept tripping over the same link.

Call it AFTER any pkill sweep. A driver that is still alive is a legitimate
owner of the link and must not have it pulled out from under it.
"""

import re
import subprocess

# BLE name the driver connects to, set on its side in ar_teleop.py as
# addSystem("PiezoLeader1", 4). Matched by NAME, never by address: the sensor
# advertises a random BLE address that changes when it reboots.
PIEZENSE_BLE_NAME = 'PiezoLeader1'

# bluetoothctl interleaves coloured [NEW]/[CHG]/[DEL] event lines with the list
# it was asked for, so its output needs stripping before it can be parsed.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def release_stale_piezense_ble(name: str = PIEZENSE_BLE_NAME) -> None:
    """Drop a BLE link to the piezense that no driver process owns any more.

    Never fatal. A missing or wedged bluetoothctl leaves the link alone and the
    session proceeds; the worst case is the behaviour we already had.
    """
    try:
        out = subprocess.run(['bluetoothctl', 'devices', 'Connected'],
                             capture_output=True, text=True, timeout=6).stdout
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return

    for raw in out.splitlines():
        line = _ANSI_RE.sub('', raw).strip()
        if not line.startswith('Device '):
            continue                       # an event line, not a list entry
        parts = line.split(maxsplit=2)     # 'Device', <mac>, <name>
        if len(parts) != 3 or parts[2].strip() != name:
            continue
        mac = parts[1]
        try:
            subprocess.run(['bluetoothctl', 'disconnect', mac],
                           capture_output=True, text=True, timeout=10)
            print(f'Released stale Bluetooth link to {name} ({mac}) — it was '
                  f'connected with no driver running')
        except (OSError, subprocess.SubprocessError):
            print(f'Could not release the Bluetooth link to {name} ({mac}). '
                  f'If the driver sits on "[example] connecting...", run:\n'
                  f'    bluetoothctl disconnect {mac}')
