import sys
sys.path.append('/home/renth/mpc_ws/src/demo1/follow_traj_re/follow_traj_re')
import rclpy
from rclpy.node import Node
from canuse import CanUse
from std_msgs.msg import Float32MultiArray
import sys

class ISGSpeedFilter:
    def __init__(self):
        self.isg_sum_filtspd = 0  # 总和
        self.isg_mot_spd_filt = 0  # 滤波后的速度
        self.isg_mot_spd = 0  # 实时速度
        self.MAT_Moto_spd = 0  # 最终输出的速度

    def update_speed(self, isg_mot_spd):
        self.isg_mot_spd = isg_mot_spd
        self.isg_sum_filtspd += self.isg_mot_spd  # 加上当前速度
        self.isg_sum_filtspd -= self.isg_mot_spd_filt  # 减去上一次的滤波结果
        
        # 计算滤波后的速度
        self.isg_mot_spd_filt = self.isg_sum_filtspd / 15
        self.MAT_Moto_spd = self.isg_mot_spd_filt  # 更新最终输出速度

        return self.MAT_Moto_spd

class CanNode(Node):
    def __init__(self):
        super().__init__('can_node')
        self.speed_filter = ISGSpeedFilter()
        self.mod_AE = 1
        self.mod_666 = 1
        self.manual_triggered = False
        self.can_use = CanUse(zone=49)  # 初始化 CanUse 类
        self.publisher_ = self.create_publisher(Float32MultiArray, 'vehicle_state', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'planner_action',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.latest_msg = None
        self.new_frame = [0, 0, 0]
        self.di = 0

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.action_timer = self.create_timer(0.05, self.publish_frame)

    def timer_callback(self):
        self.can_use.read_ins_info()
        msg = Float32MultiArray()
        print(self.can_use.ego_x)
        msg.data = [self.can_use.ego_x, 
                    self.can_use.ego_y, 
                    self.can_use.ego_v, 
                    self.can_use.ego_yaw]
        self.publisher_.publish(msg)

    def listener_callback(self, msg):
        self.latest_msg = msg
        action = self.latest_msg.data
        ai = action[0]
        self.di = action[1]
        self.di = self.speed_filter.update_speed(self.di)
        
    def publish_frame(self):
        # if self.can_use.eps_mode != 3 and self.manual_triggered:
        #     self.mod_AE = 1
        #     self.mod_666 = 1
        # if self.can_use.eps_mode == 3:
        #     self.mod_AE = 3
        #     self.mod_666 = 0
        #     self.manual_triggered = False
        if self.mod_AE == 1 and self.mod_666 == 1:
            if self.can_use.ego_x is not None and self.can_use.ego_y is not None:
                self.new_frame = [5, self.di, 0]
            else:
                print("主车定位丢失...")
                self.new_frame = [0, 0, 0]
        self.can_use.publish_planner_action(
            action = self.new_frame,
            id = 0x600,
            action_type="acc",
            mod=1,
            enable=1
        )

def main(args=None):
    rclpy.init(args=args)
    can_node = CanNode()
    rclpy.spin(can_node)
    can_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()