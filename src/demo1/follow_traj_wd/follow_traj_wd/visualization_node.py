#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from collections import deque

from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Int32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

import math
import re

def utm2localxy(state, point_x, point_y):
    det_x = point_x - state[0]
    det_y = point_y - state[1]
    distance = math.sqrt(det_x ** 2 + det_y ** 2)
    angle_line = math.atan2(det_y, det_x)
    angle = (angle_line - state[2] + math.pi / 2)
    new_x = distance * math.cos(angle)
    new_y = distance * math.sin(angle)
    return new_x, new_y, angle

class RvizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')

        self.marker_pub_ = self.create_publisher(MarkerArray, 'obstacle_markers', 1)
        self.traj_marker_pub_ = self.create_publisher(Marker, 'mpc_trajectory_marker', 1)
        self.target_marker_pub_ = self.create_publisher(Marker, 'target_point_marker', 1)
        self.prediction_traj_pub_ = self.create_publisher(Marker, 'prediction_trajectory_marker', 1)     
        self.target_points_pub_ = self.create_publisher(Marker, 'target_points_marker', 1)
        self.history_traj_pub_ = self.create_publisher(Marker, 'history_trajectory_marker', 1)
        
        self.current_trajectory = []
        self.state_queue = deque(maxlen=200)
        self.state = [0, 0, 0]
        
        self.vs_subscription = self.create_subscription(
            Float32MultiArray,
            'vehicle_state',
            self.vs_callback,
            1
        )
        
        self.subscription_mpc_trajectory = self.create_subscription(
            PoseArray,
            'mpc_trajectory',
            self.mpc_trajectory_callback,
            1
        )

        self.subscription_obstacles = self.create_subscription(
            String,
            'lidar_detections',
            self.obstacle_callback,
            1
        )
        
        self.subscription_target_point = self.create_subscription(
            Int32,
            'target_point_index',
            self.target_point_callback,
            1
        )
        
        self.subscription_prediction_trajectory = self.create_subscription(
            Float32MultiArray,
            'prediction_trajectory',
            self.prediction_trajectory_callback,
            1
        )
        
        self.subscription_target_points = self.create_subscription(
            PoseArray,
            'target_points',
            self.target_points_callback,
            1
        )

        self.marker_id_counter = 0
        self.timer = self.create_timer(0.05, self.publish_vehicle_model)
        
    def vs_callback(self, msg: Float32MultiArray):
        self.state = msg.data
        self.state_queue.append(self.state)
        self.visualize_state_queue()
    
    def visualize_state_queue(self):
        # 创建一个 Marker 用于显示状态队列轨迹
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "state_trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # 线条宽度
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # 将队列中的状态点转换为 RVIZ 可视化点
        for state in self.state_queue:
            pt = Point()
            x, y, _ = utm2localxy(self.state, state[0], state[1])
            pt.x = x
            pt.y = y
            pt.z = 0.0  # 假设 z 坐标为 0
            marker.points.append(pt)

        # 发布 Marker
        self.history_traj_pub_.publish(marker)
    
    
    def publish_vehicle_model(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vehicle_model"
        marker.id = 1000
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 车辆中心位于原点
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.75  # 高度为车高的一半

        marker.scale.x = 4.0  # 长
        marker.scale.y = 2.0  # 宽
        marker.scale.z = 1.5  # 高

        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5  # 灰色
        self.traj_marker_pub_.publish(marker)
    
    def prediction_trajectory_callback(self, msg: Float32MultiArray):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'prediction_trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0
        marker.color.r = 2.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        
        n = int(len(msg.data) / 2)
        print(msg.data)
        print(n)
        for i in range(n):
            pt = Point()
            print(msg.data[i], msg.data[n+i])
            x, y, _ = utm2localxy(self.state, msg.data[i], msg.data[n+i])
            print(self.state, x, y)
            pt.x = x
            pt.y = y
            marker.points.append(pt)

        self.prediction_traj_pub_.publish(marker)
    
    def target_points_callback(self, msg: Float32MultiArray):
        marker = Marker()
        marker.header = msg.header
        marker.ns = "target_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for pose in msg.poses:
            pt = pose.position
            new_pt = Point()
            x, y, _ = utm2localxy(self.state, pt.x, pt.y)
            new_pt.x = x
            new_pt.y = y
            marker.points.append(new_pt)

        self.target_points_pub_.publish(marker)
    
    def target_point_callback(self, msg: Int32):
        index = msg.data
        if index < 0 or index >= len(self.current_trajectory):
            self.get_logger().warn(f"Target index {index} out of range!")
            return
        pose = self.current_trajectory[index]
        pose.position.x, pose.position.y, _ = utm2localxy(self.state, pose.position.x, pose.position.y)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_point"
        marker.id = 999  # 保持唯一
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose = pose

        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  # 蓝色

        self.target_marker_pub_.publish(marker)
        self.get_logger().info(f"Published target marker at index {index}")

    def mpc_trajectory_callback(self, msg: PoseArray):
        self.current_trajectory = msg.poses
        marker = Marker()
        # marker.header = msg.header
        marker.header.frame_id = "map"  # 显式设置 frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        # 设定一个参考点（比如起点）
        for pose in msg.poses:
            pt = pose.position
            new_pt = Point()
            x, y, _ = utm2localxy(self.state, pt.x, pt.y)
            new_pt.x = x
            new_pt.y = y
            marker.points.append(new_pt)
        
        self.get_logger().info(f"Publishing marker with {len(marker.points)} points.")
        self.get_logger().info(f"Marker header frame_id: {marker.header.frame_id}")
        
        self.traj_marker_pub_.publish(marker)

    def obstacle_callback(self, msg: String):
        data = msg.data.strip()
        if not data:
            return

        # lidar_detections: x,y,z,w,l,h,yaw;...
        if data.endswith(";"):
            data = data[:-1]
        data = data.split(":")[-1]

        marker_array = MarkerArray()
        self.marker_id_counter = 0

        for box_str in data.split(";"):
            parts = box_str.strip().split(",")
            if len(parts) < 3:
                continue

            try:
                x = float(parts[0])
                y = -float(parts[1])  # 右侧为正方向
                z = float(parts[2]) if len(parts) >= 3 else 0.0
                w = float(parts[3]) if len(parts) >= 4 else 1.0
                l = float(parts[4]) if len(parts) >= 5 else 1.0
                h = float(parts[5]) if len(parts) >= 6 else 1.0

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "obstacles"
                marker.id = self.marker_id_counter
                self.marker_id_counter += 1
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = h / 2.0
                marker.scale.x = l
                marker.scale.y = w
                marker.scale.z = h
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                marker_array.markers.append(marker)

            except ValueError:
                continue

        self.marker_pub_.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RvizVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
