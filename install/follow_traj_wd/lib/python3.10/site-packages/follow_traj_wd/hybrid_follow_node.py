import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import time
import sys
sys.path.append('/home/nvidia/vcii/hezi_ros2/src/demo1/follow_traj_wd/follow_traj_wd')
from follow_demo_2025 import VehicleTrajectoryFollower
from follow_demo_mpc_bank import VehicleTrajectoryFollower as MPCVehicleTrajectoryFollower

from utils import ISGSpeedFilter
from can_use import Can_use
import logging
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
import math
import csv  
from pyproj import Proj
import numpy as np 

lonlat2xy_old = Proj('+proj=tmerc +lon_0=118.8170043 +lat_0=31.8926311 +ellps=WGS84')
def read_csv(csv_file_path):
    traj_data = []
    x = []
    y = []
    # 打开CSV文件并读取内容  
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader:  
            # 将每一行的数据转换为整数或浮点数（根据具体情况选择）  
            # 假设x坐标、y坐标、航向角和速度都是浮点数  
            x_coord = float(row[0])  
            y_coord = float(row[1])
            x_coord_utm, y_coord_utm = lonlat2xy_old(x_coord, y_coord, inverse=False)  
            heading = float(row[2])  
            x.append(x_coord_utm)
            y.append(y_coord_utm)
            # 将这些信息存储为一个列表，并添加到data_list中  
            data_row = [x_coord, y_coord, heading]  
            traj_data.append(data_row)
    return traj_data  


class HybridFollowNode(Node):
    def __init__(self):
        super().__init__('HybridFollowNode')
        self.publisher_ = self.create_publisher(
            Float32MultiArray, 
            'planner_action', 
            1
        )
        self.vs_subscription = self.create_subscription(
            Float32MultiArray,
            'vehicle_state',
            self.vs_callback,
            1
        )
        self.eps_subscription = self.create_subscription(
            Int32,
            'eps_mode',
            self.eps_callback,
            1
        )
        
        # 订阅到轨迹
        self.trajectory = self.create_subscription(
            PoseArray,
            'trajectory',
            self.trajectory_callback,
            10
        )
        
        # 订阅mpc轨迹
        self.mpc_sub_trajectory = self.create_subscription(
            PoseArray,
            'mpc_trajectory',
            self.trajectory_mpc_callback,
            10
        )
        
         # 订阅到轨迹
        self.reduce_speed = self.create_subscription(
            String,
            'obstacle_reduce_speed',
            self.obstacle_reduce_speed_callback,
            10
        )
        
        # 用于存储转换后的轨迹点数据 [[x, y, heading], ...]
        self.traj_data = []
        
        self.manual_triggered = True
        self.mode_AE = 1
        self.mode_666 = 1
        self.eps_subscription
        self.vs_subscription  # prevent unused variable warning
        self.obstacle_reduce_speed = False
        self.is_not_use_mpc = False
        self.ck = None #用于判定是否需要切换跟踪算法，根据曲率
    

        self.follower = VehicleTrajectoryFollower()
        self.MPCfollower = MPCVehicleTrajectoryFollower()
        self.filter = ISGSpeedFilter()
        self.latest_eps_mode = None
        self.can_use = Can_use()
        
    def eps_callback(self, msg):
        self.latest_eps_mode = msg.data
    
    
    def vs_callback(self, msg):
        for _ in range(20):
            self.can_use.read_ins_info()
            
        if self.can_use.ego_v >= 15:
            self.is_use_mpc = False
            self.get_logger().info(f"------------------------------------using my method track trajectory----------------------------")
            self.my_method(msg) 
        else:
            self.is_use_mpc = True
            self.get_logger().info(f"+++++++++++++++++++++++++++++++++++++using mpc track trajectory++++++++++++++++++++++++++++++++++")
            self.mpc_compute_and_publish(msg)


        
    def my_method(self, msg):
        if self.latest_eps_mode is None:
            self.get_logger().warn("尚未接收到eps_mode，跳过一次控制")
            return
        # self.get_logger().info(f"[vs_callback] Received state: {msg.data}")
        # self.get_logger().info(f"[vs_callback] EPS mode: {self.latest_eps_mode}")
        eps_mode = self.latest_eps_mode
        start = time.time()
        for i in range(20):
            self.can_use.read_ins_info()
        ego_lat = self.can_use.ego_lat
        ego_lon = self.can_use.ego_lon
        ego_yaw = self.can_use.ego_yaw_deg
        ego_v   = self.can_use.ego_v
        
        
        if eps_mode != 3 and self.manual_triggered:
            self.mode_AE = 1
            self.mode_666 = 1
        if eps_mode ==3:
            self.mode_AE = 3
            self.mode_666 = 0
            self.manual_triggered = False
        if self.mode_AE == 1 and self.mode_666 == 1:
            if ego_lon is not None and ego_lat is not None:
                turn_angle = self.follower.calculate_turn_angle(
                    (ego_lat, ego_lon, ego_yaw), ego_yaw)
                if turn_angle == "no_current_trajectory":
                    self.get_logger().info(f"没有轨迹 or 轨迹不可通行，需要停车")
                    # 需要紧急停车
                    self.frame = [float(0),    # 分别是期望速度、角度、减速度等级
                                float(0), 
                                float(-3)]
                else:
                    filtered_angle = self.filter.update_speed(turn_angle)
                    desired_speed, desired_acc = self.follower.calculate_speedAndacc(
                            turn_angle, (ego_lat, ego_lon, ego_yaw), ego_v, is_obstacle = self.obstacle_reduce_speed)
                    # logging.info(f'trun angle: {turn_angle}, filter angle: {filtered_angle}')
                    self.frame = [float(desired_speed), 
                                float(filtered_angle), 
                                float(desired_acc)]
                    
                    
                planner_frame = Float32MultiArray()
                planner_frame.data = self.frame
                # self.get_logger().info(f"[vs_callback] Send frame: {planner_frame.data}")
                self.publisher_.publish(planner_frame)
        elapsed_time = time.time() - start
        # self.get_logger().info(f"calc time:{elapsed_time:.6f} [sec]")

    def mpc_compute_and_publish(self,msg):
        """
        手动调用的计算与发布函数：
        1. 读取当前车辆状态（通过 can_use 对象）
        2. 调用 MPCfollower 计算控制指令
        3. 对控制指令进行滤波
        4. 发布最新计算得到的控制指令
        """
        # start_time = time.time()
        
        for _ in range(20):
            self.can_use.read_ins_info()
        ego_lat = self.can_use.ego_lat
        ego_lon = self.can_use.ego_lon
        ego_yaw = self.can_use.ego_yaw_deg
        ego_v   = self.can_use.ego_v

        new_frame = [0,0,0]
        # 调用 MPCfollower 计算控制指令
        if self.can_use.ego_x is not None and self.can_use.ego_y is not None and (self.MPCfollower.cx is not None):     
            turn_angle = self.MPCfollower.calculate_turn_angle((self.can_use.ego_x, self.can_use.ego_y, self.can_use.ego_yaw_rad), self.can_use.ego_yaw_rad, self.can_use.ego_v)
            # turn_angle = 0
            
            # 因为内部有计算参考轨迹的时候，会计算目标点，这里就不再计算了 
            self.MPCfollower.update_target_index((self.can_use.ego_x, self.can_use.ego_y), self.can_use.ego_yaw_rad, self.can_use.ego_v)
            
            # print("turn_angle=", turn_angle)
            filtered_angle = self.filter.update_speed(turn_angle)
            desired_speed, desired_acc = self.MPCfollower.calculate_speedAndacc(
                            turn_angle, (ego_lat, ego_lon, ego_yaw), ego_v, is_obstacle = self.obstacle_reduce_speed)
            # print("===========",turn_angle)
            # print("time cost: ",time.time()-t0)
            new_frame = [float(desired_speed),
                         float(filtered_angle), 
                         float(desired_acc)]     
            # new_frame = [float(5),
            #              float(0), 
            #              float(0)] 
        else:
            self.get_logger().info(f"self.follower.cx is None")
            new_frame = [0.0, 0.0, -3.0]     
        
        # 构造发布消息（示例中使用固定加速度5.0，实际可根据需求调整）
        planner_frame = Float32MultiArray()
        planner_frame.data = new_frame
        
        # 发布消息
        # self.get_logger().info(f"Publishing frame: {planner_frame.data}")
        self.publisher_.publish(planner_frame)
        
        # elapsed_time = time.time() - start_time
        # self.get_logger().info(f"Calculation time: {elapsed_time:.6f} sec")


    def trajectory_callback(self, msg: PoseArray):
        """
        将订阅到的 PoseArray 转换为和 CSV 中相同的格式：
        [[x_utm, y_utm, heading], [x_utm, y_utm, heading], ...]
        """
        traj_data = []
        # 为空，表示没有轨迹或者轨迹不可用，需要停止
        if len(msg.poses) == 0:
            # print("len of msg.poses: " , len(msg.poses))
            self.follower.current_trajectory = None
        else:
            for pose in msg.poses:
                x = pose.position.x
                y = pose.position.y
                
                # 提取四元数 (只关心 z, w，假设 x=y=0)
                qz = pose.orientation.z
                qw = pose.orientation.w
                
                # 反向获取 yaw（弧度）
                yaw_rad = 2.0 * math.atan2(qz, qw)
                # 将弧度转为度数
                heading_deg = math.degrees(yaw_rad)
                # 归一化到 [0, 360)
                heading_deg = (heading_deg + 360) % 360
                traj_data.append([x, y, heading_deg])
                
            # 更新最新的轨迹
            self.follower.current_trajectory = traj_data

 
    def trajectory_mpc_callback(self, msg: PoseArray):
        """
        将订阅到的 PoseArray 转换为和 CSV 中相同的格式：
        [x_utm, y_utm, yaw_rad, ck]

        注意：此用法要求发布端在 Pose.orientation.x 中放置了 ck，
            orientation.z / orientation.w 用于 Z 轴 yaw。
        """
        ref_x = []
        ref_y = []
        ref_yaw = []
        ref_k = []
        self.get_logger().info(f"subscribe mpc trajectory")
        # 若消息为空，则认为无轨迹或轨迹不可用
        if len(msg.poses) == 0:
            # print("len of msg.poses:", len(msg.poses))
            self.follower.current_trajectory = None
        else:
            for pose in msg.poses:
                x = pose.position.x
                y = pose.position.y

                # 在发布时把曲率 ck 塞进 orientation.x (非标准用法)
                ck = pose.orientation.x

                # 从 orientation.z, orientation.w 解出 yaw（弧度）
                qz = pose.orientation.z
                qw = pose.orientation.w
                yaw_rad = 2.0 * math.atan2(qz, qw)
                # 此处 yaw_rad 范围在 (-pi, pi)，如果你想让它一直正向增长，
                # 可以再做一次归一化或转换，但通常原样就够用了。

                # 将结果存入列表： [x, y, yaw_rad, ck]
                ref_x.append(x)
                ref_y.append(y)
                ref_yaw.append(yaw_rad)
                ref_k.append(ck)

            # 将解析后的轨迹保存到你的 follower 中
            
            self.MPCfollower.cx   = ref_x
            self.MPCfollower.cy   = ref_y
            self.MPCfollower.cyaw = ref_yaw
            self.MPCfollower.ck   = ref_k
            self.ck               = ref_k


    def obstacle_reduce_speed_callback(self, msg):
        data = str(msg.data)
        if data == 'Yes':
            self.obstacle_reduce_speed = True
        else:
            self.obstacle_reduce_speed = False
        self.get_logger().info(f"是否因为障碍物减速：{data}")
    

    # 判断是否是弯道
    def is_curve(self, target_ind, threshold = 0.045):
        curvature = self.cal_curvature(target_ind)
        if curvature > threshold:
            return True
        else:
            return False
     
    # 计算未来一段路的曲率
    def cal_curvature(self,target_ind, num_point = 20):
        '''返回一段路的曲率'''
        threshold = 0.05
        # Check if start_index is valid
        start_index = target_ind
        end_index = min(start_index + 20, len(self.ck))
        sublist = self.ck[start_index:end_index]
        print("sublist :",sublist, "target_ind: ",target_ind)
        sublist = list(map(abs, sublist)) # 统一转为绝对值
        print("np.mean(sublist): ",np.mean(sublist))
        return np.mean(sublist)


def main(args=None):
    rclpy.init(args=args)
    follow_node = HybridFollowNode()
    rclpy.spin(follow_node)
    FollowNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()