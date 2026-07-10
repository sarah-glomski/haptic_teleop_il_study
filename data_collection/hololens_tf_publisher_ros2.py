#!/usr/bin/env python3
"""
HoloLens TF Publisher — ROS2 port of teleop-haptics_study/scripts/hololens_tf_publisher.py

Bridges HoloLens 2 hand tracking data (arriving via rosbridge_websocket) to the
ROS2 TF tree. The HoloLens Unity app connects to the rosbridge WebSocket server
(default port 9090) and publishes individual joint poses as PoseStamped.

Subscribed topics (from HoloLens via rosbridge):
  /hololens/palm/right       geometry_msgs/PoseStamped
  /hololens/palm/left        geometry_msgs/PoseStamped
  /hololens/thumb/right      geometry_msgs/PoseStamped
  /hololens/index/right      geometry_msgs/PoseStamped
  /hololens/gaze             geometry_msgs/PoseStamped
  /hololens/virtual_base_link geometry_msgs/PoseStamped

TF tree broadcasts (parent → child):
  Unity → right_palm
  Unity → left_palm
  Unity → right_thumb
  Unity → right_index
  Unity → gaze
  Unity → virtual_base_link
  virtual_base_link → base_link  (static identity)

Usage:
  ros2 run haptic_teleop_il_study hololens_tf_publisher_ros2
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


class HoloLensTFPublisher(Node):
    """
    ROS2 node that republishes HoloLens joint poses as TF2 transforms.
    Direct port of the ROS1 HoloLensFramePublisher to ROS2.
    """

    # Map from subscribed topic suffix → TF child frame ID
    _TOPIC_FRAME_MAP = {
        '/hololens/palm/right':       'right_palm',
        '/hololens/palm/left':        'left_palm',
        '/hololens/thumb/right':      'right_thumb',
        '/hololens/index/right':      'right_index',
        '/hololens/gaze':             'gaze',
        '/hololens/virtual_base_link': 'virtual_base_link',
    }

    def __init__(self):
        super().__init__('hololens_tf_publisher')

        self._br = TransformBroadcaster(self)
        self._static_br = StaticTransformBroadcaster(self)

        # Subscribe to each HoloLens topic
        for topic, frame_id in self._TOPIC_FRAME_MAP.items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, fid=frame_id: self._broadcast_tf(msg, fid),
                10,
            )

        # Publish static identity transform: virtual_base_link → base_link
        self._broadcast_static_identity()

        self.get_logger().info('HoloLens TF Publisher (ROS2) initialized — '
                               'waiting for HoloLens via rosbridge on port 9090')

    def _broadcast_tf(self, msg: PoseStamped, child_frame: str):
        """Convert a PoseStamped from the HoloLens into a TF2 transform broadcast."""
        # Skip zero poses — HoloLens sends zeros when a joint is not tracked
        p = msg.pose.position
        if p.x == 0.0 and p.y == 0.0 and p.z == 0.0:
            return

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'Unity'
        t.child_frame_id = child_frame

        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = msg.pose.orientation

        self._br.sendTransform(t)

    def _broadcast_static_identity(self):
        """Publish a static identity transform: virtual_base_link → base_link.

        This mirrors the publish_static_transform() in the original ROS1 node.
        In practice, if the virtual_base_link is correctly placed in the HoloLens
        world to coincide with the robot's physical base, this identity is valid.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'virtual_base_link'
        t.child_frame_id = 'base_link'
        # Identity rotation
        t.transform.rotation.w = 1.0
        self._static_br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = HoloLensTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
