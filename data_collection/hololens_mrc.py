#!/usr/bin/env python3
"""
HoloLens Mixed Reality Capture client — per-episode first-person video.

Talks to the HoloLens Device Portal REST API to start/stop MRC video
recordings and download the resulting MP4s. Used by hdf5_data_collector.py to
record what the wearer sees (real world + holograms, incl. the gaze cursor)
for each episode, saved as demo_data/episode_N.mp4 next to the HDF5.

Credentials live in hololens_portal.json next to this file (gitignored):
  {"ip": "172.24.95.144", "user": "...", "password": "..."}

Portal API quirks handled here (verified against the device):
  - POSTs need a Content-Length (empty body) and an X-CSRF-Token header that
    echoes the CSRF-Token cookie from any prior GET.
  - Recording can only start while the headset is awake/worn; a start failure
    raises and the caller should continue without video.
  - File download/delete take the filename base64-encoded.
"""

import base64
import json
import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CRED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'hololens_portal.json')


class MRCClient:

    def __init__(self):
        if not os.path.isfile(CRED_FILE):
            raise FileNotFoundError(
                f'{CRED_FILE} not found — create it with '
                '{"ip": ..., "user": ..., "password": ...}')
        with open(CRED_FILE) as f:
            creds = json.load(f)
        self.base = f"https://{creds['ip']}/api/holographic/mrc"
        self.s = requests.Session()
        self.s.auth = (creds['user'], creds['password'])
        self.s.verify = False
        self._start_marker = 0   # newest CreationTime before recording began

    # ── low-level ─────────────────────────────────────────────────────────────
    def _refresh_csrf(self):
        self.s.get(f'{self.base}/status', timeout=5)
        tok = self.s.cookies.get('CSRF-Token')
        if tok:
            self.s.headers['X-CSRF-Token'] = tok

    def _files(self):
        r = self.s.get(f'{self.base}/files', timeout=10)
        r.raise_for_status()
        return r.json().get('MrcRecordings', [])

    def _file_param(self, filename):
        return {'filename': base64.b64encode(filename.encode()).decode()}

    def _delete(self, filename):
        self.s.delete(f'{self.base}/file', params=self._file_param(filename),
                      timeout=15)

    # ── recording control ─────────────────────────────────────────────────────
    def start(self):
        """Begin an MRC recording (holograms + PV camera, no audio)."""
        self._refresh_csrf()
        self._start_marker = max((f['CreationTime'] for f in self._files()),
                                 default=0)
        r = self.s.post(
            f'{self.base}/video/control/start',
            params={'holo': 'true', 'pv': 'true',
                    'mic': 'false', 'loopback': 'false'},
            data=b'', timeout=10)
        if r.status_code != 200:
            reason = ''
            try:
                reason = r.json().get('Reason', '')
            except Exception:
                pass
            raise RuntimeError(
                f'MRC start failed (HTTP {r.status_code} {reason}) — '
                'is the headset awake/worn?')

    def stop(self):
        self._refresh_csrf()
        r = self.s.post(f'{self.base}/video/control/stop', data=b'', timeout=10)
        if r.status_code not in (200, 204):
            raise RuntimeError(f'MRC stop failed (HTTP {r.status_code})')

    def _new_files(self):
        return [f for f in self._files()
                if f['CreationTime'] > self._start_marker]

    # ── high-level episode operations ─────────────────────────────────────────
    def stop_and_fetch(self, dest_path, timeout_s=60):
        """Stop recording, wait for the MP4 to appear, download it to
        dest_path, delete it from the device. Returns (dest_path, bytes)."""
        self.stop()
        newest = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            new = self._new_files()
            if new:
                newest = max(new, key=lambda f: f['CreationTime'])
                break
            time.sleep(1.0)
        if newest is None:
            raise RuntimeError('no new MRC file appeared after stop')

        # Give the device a moment to finalize the file before download.
        time.sleep(2.0)
        r = self.s.get(f'{self.base}/file',
                       params=self._file_param(newest['FileName']),
                       timeout=300, stream=True)
        r.raise_for_status()
        tmp = dest_path + '.part'
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(tmp, dest_path)
        self._delete(newest['FileName'])
        return dest_path, os.path.getsize(dest_path)

    def stop_and_discard(self):
        """Stop recording and delete the resulting file (cancelled episode)."""
        self.stop()
        time.sleep(2.0)
        for f in self._new_files():
            self._delete(f['FileName'])
