#!/usr/bin/env python3
"""Keep the Piezense in the configuration data collection needs.

THE PROBLEM (found 2026-08-22): piezense_ros/ar_teleop.py fires the device
configuration over BLE the instant the link opens — actuator mode + 110 kPa
setpoint on all four channels, and SENSE mode (+ correction 15, P=5) on
channels 2/3, the gripper pads this study records. If the device is not ready
for those first writes they are silently lost: ch2/3 stay in actuator mode,
their PID loop regulates every grasp back out, and they rest at ~117 kPa
instead of the 110 kPa setpoint. That is what "the piezense isn't working"
looks like — a +4 kPa blip that decays under a held grip instead of a +17-20
kPa step that holds. It poisoned episodes 0-61 of Task2Collection1 and a fresh
relaunch after a reboot reproduced it; power-cycling and re-plugging the air
did nothing. Re-sending the same configuration over the driver's own
/piezense/config + /piezense/setpoint topics fixed it within a second.

This script does not touch the piezense package. It talks to the running
driver exactly the way ar_teleop.py does, from outside.

    python3.12 piezense_reconfig.py            # one-shot: re-send the config now
    python3.12 piezense_reconfig.py --watch    # node: send the config once per BLE
                                               # link, then keep watching for the
                                               # wrong-mode baseline
                                               # (launch_data_collection starts this)

WHY IT SENDS UNCONDITIONALLY. The baseline test below catches only the total
failure. A partial apply — sense mode on, but set_sense_correction_value or the
ch2/3 P gain missing — rests at a healthy ~109 kPa and still reports pressure,
just ~25% too little. That is invisible live and only shows up when the
recorded demos are compared against earlier sessions (2026-08-25: 5.04 kPa per
unit gripper closure against 6.85 and 6.82 in the two prior sessions, at
identical closure). Sending the whole config once per link is idempotent and
removes the failure mode instead of trying to detect it.

DETECTING THE WRONG MODE without knowing whether the operator is gripping:
use the MIN of ch2/3 over a few seconds (their resting floor) with both bounds:
    healthy at rest   min ~109-111
    healthy gripping  min ~125-131   (above the upper bound -> not flagged)
    wrong mode        min ~114-121   (rest 116-119, grip blips decay back)
so 113 < min < 123 means "wrong mode", and a held healthy grip cannot fake it.
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
    def __init__(self):
        super().__init__('piezense_reconfig_watch')
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
        self.get_logger().info('watching piezense/data — will re-send the '
                               'configuration if ch2/3 rest in the wrong mode')

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

        # Send the whole configuration ONCE per link, whatever the baseline
        # says. The baseline test below only catches the total failure (ch2/3
        # left in actuator mode, resting ~117); a PARTIAL apply — sense mode on
        # but set_sense_correction_value / the ch2/3 P gain missing — leaves the
        # baseline at a healthy ~109 while the sensitivity is down by a quarter.
        # That is invisible here and only shows up in the recorded data: on
        # 2026-08-25 five grape demos read 5.04 kPa per unit of gripper closure
        # where the two previous sessions both read 6.8, at identical closure.
        # Re-sending is idempotent, so do it rather than try to detect it.
        if not self._primed:
            self._primed = True
            self.get_logger().info(
                'priming the piezense configuration for this link (~8 s, '
                'includes the settling waits the sense calibration needs)')
            send_configuration(self._cfg, self._sp, log=self.get_logger().info)
            self._last_send_t = now
            self._samples.clear()
            return
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
        if self._sends_this_link >= MAX_SENDS_PER_LINK:
            self.get_logger().error(
                f'piezense STILL in wrong mode after {MAX_SENDS_PER_LINK} '
                f'config sends (floor {floor:.1f} kPa) — relaunch the driver')
            self._last_send_t = now
            return
        self.get_logger().warn(
            f'piezense ch{SENSE_CHANNELS} resting at {floor:.1f} kPa (wrong '
            f'mode, want ~110) — re-sending configuration '
            f'[{self._sends_this_link + 1}/{MAX_SENDS_PER_LINK}]')
        send_configuration(self._cfg, self._sp, log=self.get_logger().info)
        self._last_send_t = now
        self._sends_this_link += 1
        self._samples.clear()
        self._link_ok_logged = False


def main():
    rclpy.init()
    if '--watch' in sys.argv:
        node = PiezenseWatch()
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
    cfg, sp = _publishers(node)
    time.sleep(1.0)               # discovery
    send_configuration(cfg, sp)
    time.sleep(0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
