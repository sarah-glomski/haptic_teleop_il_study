#!/usr/bin/env python3
"""
Kinova Gen3 State Publisher Node (ROS2)

Reads current robot state from the Kinova Gen3 via the Kortex Python API and
publishes it at 30 Hz. This node is READ-ONLY and sends no commands.

Published topics:
  robot_obs/pose         geometry_msgs/PoseStamped  - current TCP pose (base_link frame)
  robot_obs/gripper      std_msgs/Float32            - gripper position [0=open, 1=closed]
  robot_obs/joint_states sensor_msgs/JointState      - joint angles (radians)

ROS2 Parameters:
  robot_ip   str   '192.168.1.10'  Kinova Gen3 IP address
  username   str   'admin'
  password   str   'admin'
  num_joints int   7               DOF count (7 for Gen3)

Dependencies:
  pip install kortex-api
  (or install from https://github.com/Kinovarobotics/kortex/releases)

Usage:
  ros2 run haptic_teleop_il_study kinova_state_publisher
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2


class KinovaStatePublisher(Node):
    """
    Read-only ROS2 node that polls the Kinova Gen3 Kortex API and publishes
    current robot state (TCP pose, gripper, joint angles) at 30 Hz.

    Mirrors xarm_state_publisher.py from Robomimic/data_collection but uses
    the Kortex SDK instead of the xArm SDK.
    """

    def __init__(self):
        super().__init__('kinova_state_publisher')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.robot_ip  = self.declare_parameter('robot_ip',   '192.168.1.10').value
        self.username  = self.declare_parameter('username',   'admin').value
        self.password  = self.declare_parameter('password',   'admin').value
        self.num_joints = self.declare_parameter('num_joints', 7).value

        # ── Connect to robot ────────────────────────────────────────────────────
        self.get_logger().info(f'Connecting to Kinova Gen3 at {self.robot_ip} …')
        self._connect()
        self.get_logger().info('Connected to Kinova Gen3')

        # ── Publishers ──────────────────────────────────────────────────────────
        self.pose_pub   = self.create_publisher(PoseStamped, 'robot_obs/pose',         10)
        self.gripper_pub = self.create_publisher(Float32,    'robot_obs/gripper',      10)
        self.joints_pub = self.create_publisher(JointState,  'robot_obs/joint_states', 10)

        # ── 30 Hz timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._publish_state)

        self.get_logger().info('Kinova State Publisher initialized')

    # ── Connection ────────────────────────────────────────────────────────────
    def _connect(self):
        self._transport = TCPTransport()
        self._router = RouterClient(
            self._transport,
            lambda ex: self.get_logger().error(f'Kortex error: {ex}'),
        )
        self._transport.connect(self.robot_ip, 10000)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000   # ms
        session_info.connection_inactivity_timeout = 2000  # ms

        self._session_manager = SessionManager(self._router)
        self._session_manager.CreateSession(session_info)

        self._base         = BaseClient(self._router)
        self._base_cyclic  = BaseCyclicClient(self._router)

    # ── State publishing ──────────────────────────────────────────────────────
    def _publish_state(self):
        try:
            # RefreshFeedback() is the fast path: one round-trip for all fields
            feedback = self._base_cyclic.RefreshFeedback()
            now = self.get_clock().now().to_msg()

            # ── TCP pose ─────────────────────────────────────────────────────
            # Kinova reports position in metres and orientation as Euler angles (deg)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = 'base_link'

            pose_msg.pose.position.x = feedback.base.tool_pose_x
            pose_msg.pose.position.y = feedback.base.tool_pose_y
            pose_msg.pose.position.z = feedback.base.tool_pose_z

            quat = R.from_euler('xyz', [
                math.radians(feedback.base.tool_pose_theta_x),
                math.radians(feedback.base.tool_pose_theta_y),
                math.radians(feedback.base.tool_pose_theta_z),
            ]).as_quat()  # [qx, qy, qz, qw]

            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            self.pose_pub.publish(pose_msg)

            # ── Joint angles ─────────────────────────────────────────────────
            js_msg = JointState()
            js_msg.header.stamp = now
            js_msg.name = [f'joint_{i + 1}' for i in range(self.num_joints)]
            js_msg.position = [
                math.radians(act.position)
                for act in list(feedback.actuators)[: self.num_joints]
            ]
            self.joints_pub.publish(js_msg)

            # ── Gripper position ──────────────────────────────────────────────
            gripper_feedback = self._base.GetMeasuredGripperMovement(
                Base_pb2.GripperRequest(mode=Base_pb2.GRIPPER_POSITION)
            )
            gripper_val = 0.0
            if gripper_feedback.finger:
                gripper_val = float(gripper_feedback.finger[0].value)  # 0=open, 1=closed

            grip_msg = Float32()
            grip_msg.data = gripper_val
            self.gripper_pub.publish(grip_msg)

        except Exception as e:
            self.get_logger().error(f'State publish error: {e}')

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy_node(self):
        self.get_logger().info('Disconnecting from Kinova Gen3 …')
        try:
            self._session_manager.CloseSession()
            self._transport.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinovaStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
