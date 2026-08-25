#!/usr/bin/env python3
"""Watch the Piezense configuration state — and do NOT try to repair it.

READ THIS BEFORE USING THE ONE-SHOT MODE. It is kept for diagnosis only.

The fault it was written for is real: piezense_ros/ar_teleop.py sends the
device configuration the instant BLE connects, and if the device is not ready
those writes are lost. Channels 2/3 (the gripper pads this study records) stay
in actuator mode, rest at ~117 kPa instead of the 110 kPa setpoint, and their
PID loop regulates every grasp away.

But this script CANNOT correctly repair it, because of how the values travel:

    piezense/__init__.py  sendConfig  ->  message = f"{key}:{value}"
    piezense_driver.py    config_callback  ->  config_value = float(paramList[3])

Everything sent over /piezense/config is cast to float before it is formatted
into the string the firmware parses. So:

    ar_teleop.py native      "set_act_mode:1"    "set_sense_correction_value:15"
    anything via this topic  "set_act_mode:1.0"  "set_sense_correction_value:15.0"

Every integer-valued parameter arrives in a different form than upstream sends.
On 2026-08-25 that produced a sensor which looked healthy — resting at 109 kPa,
correct response shape — while reading ~25% low, felt by the operator through
the haptic actuators (the launcher forwards sense ch2/ch3 into actuator ch0/ch1
via ((x-110000)*4+110000), so the feedback IS the sensor reading, x4) and
visible in the recorded demos as peak grip pressure of 10.05 kPa where two
earlier sessions read 12.50 and 13.60 at identical gripper closure.

The float cast is in the piezense package, which this study does not modify, so
there is no way to send an integer through this topic. THE FIX FOR A WRONG-MODE
PIEZENSE IS TO RESTART THE DRIVER — a fresh ar_teleop.py run sends the correct
types natively, with its own 1 s and 5 s settling waits.

    python3.12 piezense_reconfig.py --watch    # detect + alert (what the
                                               #   collection launch runs)
    python3.12 piezense_reconfig.py            # one-shot send: DIAGNOSTIC ONLY,
                                               #   leaves the sense channels
                                               #   miscalibrated
    ... --watch --send-anyway                  # send on detection anyway

Detecting the wrong mode uses the ch2/3 resting floor over a few seconds, with
both bounds so a held healthy grip cannot fake it:
    healthy at rest   min ~109-111
    healthy gripping  min ~125-131   (above the upper bound -> not flagged)
    wrong mode        min ~114-121
so 113 < min < 123 means "wrong mode".

To tell whether recorded episodes are affected, use piezense_sensitivity.py —
peak grip pressure against a reference collection.
"""


import sys
import time

import rclpy
from rclpy.node import Node

from piezense_interfaces.msg import PiezenseConfig, PiezenseSetpoint, PiezenseSystemArray

SENSE_CHANNELS = (2, 3)           # the gripper pads the collector records
ACTUATOR_CHANNELS = (0, 1)        # what the operator feels; driven BY the sense channels
FORWARD_EXPR = '((x-110000)*4+110000)'   # launcher_ar_teleop_piezense.py's mapping
SETPOINT_PA = 110000
WRONG_MODE_MIN_KPA = 113.0        # resting floor above this ...
WRONG_MODE_MAX_KPA = 123.0        # ... and below this = wrong mode
WINDOW_S = 4.0                    # resting-floor window
RECONNECT_GAP_S = 5.0             # data silence this long = BLE reconnect
SEND_COOLDOWN_S = 30.0   # the send itself now takes ~8 s (settling waits)
MAX_SENDS_PER_LINK = 8


def _publishers(node):
    return (node.create_publisher(PiezenseConfig, 'piezense/config', 10),
            node.create_publisher(PiezenseSetpoint, 'piezense/setpoint', 10))


def send_configuration(cfg_pub, sp_pub, log=print, forwarding=True):
    """Replay piezense_ros's own connect sequence, INCLUDING its waits.

    The timing is not decoration. ar_teleop.py sleeps 1 s after the ch0/ch1 PID
    block and 5 s after the setpoints before it applies
    set_sense_correction_value / set_sense_mode to ch2 and ch3, because that
    correction calibrates against a settled reading. Sending the same values
    back-to-back leaves the sensor in the right mode with the wrong gain: on
    2026-08-25 a rushed version of this function (0.15 s between every send)
    restored a healthy 109 kPa baseline but left the sense channels ~26% down,
    which the operator felt through the haptic actuators and which showed up
    in the recorded demos. Keep the sleeps.

    Forwarding is re-established afterwards because it is what makes the rig
    haptic: launcher_ar_teleop_piezense.py maps sensor ch2 -> actuator ch0 and
    ch3 -> actuator ch1 through ((x-110000)*4+110000). Re-sending modes without
    re-sending forwarding risks a configured sensor driving nothing.
    """
    def C(ch, name, val):
        m = PiezenseConfig(); m.function = 'sendConfig'
        m.parameters = f'0,{ch},{name},{val}'
        cfg_pub.publish(m); time.sleep(0.15)

    def S(ch, pa):
        m = PiezenseSetpoint(); m.system_id = 0; m.channel_id = ch
        m.pressure_pa = int(pa)
        sp_pub.publish(m); time.sleep(0.15)

    def F(sense_ch, act_ch):
        m = PiezenseConfig(); m.function = 'addForwarding'
        m.parameters = f'0,{sense_ch},0,{act_ch},{FORWARD_EXPR}'
        cfg_pub.publish(m); time.sleep(0.3)

    for ch in range(4):
        C(ch, 'set_act_mode', 1)
    for ch in ACTUATOR_CHANNELS:
        C(ch, 'set_pid_Pvalues_p', 0.8); C(ch, 'set_pid_Pvalues_i', 0.15)
        C(ch, 'set_pid_Pvalues_d', 0.0); C(ch, 'set_pid_Vvalues_p', 4.0)
        C(ch, 'set_pid_Vvalues_i', 0.3); C(ch, 'set_pid_Vvalues_d', 0.0)
    log('  settling 1 s before setpoints (as ar_teleop.py does) …')
    time.sleep(1.0)
    for ch in range(4):
        S(ch, SETPOINT_PA)
    log('  settling 5 s before the sense calibration — this wait IS the '
        'calibration condition, do not shorten it …')
    time.sleep(5.0)
    for ch in SENSE_CHANNELS:
        C(ch, 'set_sense_correction_value', 15)
    for ch in SENSE_CHANNELS:
        C(ch, 'set_pid_Pvalues_p', 5.0)
    for ch in SENSE_CHANNELS:
        C(ch, 'set_sense_mode', 1)
    if forwarding:
        time.sleep(0.5)
        for sense_ch, act_ch in zip(SENSE_CHANNELS, ACTUATOR_CHANNELS):
            F(sense_ch, act_ch)
        log(f'  forwarding re-established: sense ch{SENSE_CHANNELS} -> '
            f'actuator ch{ACTUATOR_CHANNELS} via {FORWARD_EXPR}')
    log('Piezense configuration sent (act modes, PID, setpoints, sense '
        'calibration, forwarding)')


class PiezenseWatch(Node):
    def __init__(self, may_send=False):
        super().__init__('piezense_reconfig_watch')
        self._may_send = may_send
        self._cfg, self._sp = _publishers(self)
        self.create_subscription(PiezenseSystemArray, 'piezense/data',
                                 self._data_cb, 10)
        self._samples = []            # (t, min over sense channels, kPa)
        self._primed = False          # unconditional config send done for this link
        self._last_data_t = None
        self._last_send_t = None
        self._sends_this_link = 0
        self._link_ok_logged = False
        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            'watching piezense/data — will ALERT if ch2/3 rest in the wrong '
            'mode. It will not reconfigure: config sent over the ROS topic is '
            'float-typed and miscalibrates the sense channels (see docstring). '
            'The fix is to restart the driver.'
            + ('  [--send-anyway: will send regardless]' if self._may_send else ''))

    def _data_cb(self, msg):
        now = time.monotonic()
        if self._last_data_t is not None and now - self._last_data_t > RECONNECT_GAP_S:
            self.get_logger().warn(
                f'piezense data resumed after {now - self._last_data_t:.0f}s '
                'silence — BLE reconnect, config may be lost; re-checking')
            self._samples.clear()
            self._sends_this_link = 0
            self._link_ok_logged = False
            self._primed = False        # re-prime: a new link starts unconfigured
        self._last_data_t = now
        for s in msg.system:
            p = list(s.pressure_pa)
            if len(p) > max(SENSE_CHANNELS):
                self._samples.append((now, min(p[c] for c in SENSE_CHANNELS) / 1000.0))
            break
        cutoff = now - WINDOW_S
        self._samples = [x for x in self._samples if x[0] >= cutoff]

    def _tick(self):
        if not self._samples:
            return
        now = time.monotonic()

        if self._samples[-1][0] - self._samples[0][0] < WINDOW_S - 0.5:
            return                                # window not full yet
        floor = min(v for _, v in self._samples)
        wrong = WRONG_MODE_MIN_KPA < floor < WRONG_MODE_MAX_KPA
        if not wrong:
            if not self._link_ok_logged:
                self.get_logger().info(
                    f'piezense OK — ch{SENSE_CHANNELS} resting floor {floor:.1f} kPa')
                self._link_ok_logged = True
            return
        if self._last_send_t and now - self._last_send_t < SEND_COOLDOWN_S:
            return
        if not self._may_send:
            self.get_logger().error(
                '\n' + '!' * 68 +
                f'\n  PIEZENSE IS IN THE WRONG MODE — ch{SENSE_CHANNELS} resting at '
                f'{floor:.1f} kPa, want ~110.'
                '\n  RESTART THE PIEZENSE DRIVER (relaunch data collection). Do not'
                '\n  record until the resting level reads ~109: this script can no'
                '\n  longer repair it, see the module docstring.'
                '\n' + '!' * 68)
            self._last_send_t = now
            return
        if self._sends_this_link >= MAX_SENDS_PER_LINK:
            self.get_logger().error(
                f'piezense STILL in wrong mode after {MAX_SENDS_PER_LINK} '
                f'config sends (floor {floor:.1f} kPa) — restart the driver')
            self._last_send_t = now
            return
        self.get_logger().warn(
            f'piezense ch{SENSE_CHANNELS} resting at {floor:.1f} kPa — sending '
            f'configuration ANYWAY at your request (--send-anyway); the values '
            f'arrive float-typed, see the docstring '
            f'[{self._sends_this_link + 1}/{MAX_SENDS_PER_LINK}]')
        send_configuration(self._cfg, self._sp, log=self.get_logger().info)
        self._last_send_t = now
        self._sends_this_link += 1
        self._samples.clear()
        self._link_ok_logged = False


def main():
    rclpy.init()
    if '--watch' in sys.argv:
        node = PiezenseWatch(may_send='--send-anyway' in sys.argv)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, Exception):
            pass                      # launch teardown; nothing to clean up
        finally:
            try:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
        return
    node = Node('piezense_reconfig_once')
    print('!' * 68)
    print('  DIAGNOSTIC ONLY. Values sent over /piezense/config are cast to')
    print('  float by the driver, so integer parameters reach the firmware as')
    print('  "set_sense_correction_value:15.0" instead of ":15". This leaves the')
    print('  sense channels MISCALIBRATED (~25% low) while looking healthy.')
    print('  To actually fix a wrong-mode piezense, RESTART THE DRIVER.')
    print('!' * 68)
    cfg, sp = _publishers(node)
    time.sleep(1.0)               # discovery
    send_configuration(cfg, sp)
    time.sleep(0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
