import math
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pi_2_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def read_csv(csv_file_path): 
    x_coords = []
    y_coords = []
    heading_list = []
    speed_list = []
    # 打开CSV文件并读取内容  
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader:  
            lon = float(row[0])  
            lat = float(row[1])
            heading = float(row[2])
            # 将经纬度转换为UTM坐标
            x_coords.append(lon)
            y_coords.append(lat)
            heading_list.append(heading)
            speed_list.append(float(row[3]))
            # 将UTM坐标和航向角存储到traj_data中  
    return x_coords, y_coords, heading_list, speed_list

class LaneChangeDecider():
    
    def __init__(self, path):
        self.target_speed = 10.0/3.6
        self.cx, self.cy, self.cyaw, self.ck = read_csv(path)
        self.sp = self.calc_speed_profile(self.target_speed)
        self.cyaw = self.smooth_yaw(self.cyaw)
        self.obs_list = []
        self.all_line_temp = []
        self.ref_line = []
        self.new_line = []
        self.state = None
        self.init_refline()
        self.planning = False
    
    def smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]
            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
        return yaw
    
    def init_refline(self):
        for i in range(len(self.cx)):
            self.all_line_temp.append([
                self.cx[i],
                self.cy[i],
                self.cyaw[i],
                self.ck[i],
                self.sp[i]
            ])
    
    def update_state(self, state, obs_list):
        self.state = state
        self.obs_list = obs_list
        self.update_local_refline()
        self.get_current_observation()
    
    def update_local_refline(self):
        if len(self.all_line_temp) != 0:
            for j in range(len(self.all_line_temp)):
                new_x, new_y, new_yaw = self.utm2localxy(self.state, self.all_line_temp[j][0], self.all_line_temp[j][1])
                self.ref_line.append([new_x, new_y, new_yaw, self.all_line_temp[j][3]])
    
    def get_current_observation(self):
        self.obs_list = [
        [x, y, z, pixel_cx, pixel_cy, pixel_w, pixel_h]
        for x, y, z, pixel_cx, pixel_cy, pixel_w, pixel_h in self.obs_list]
        columns = ['x', 'y', 'z', 'pixel_cx', 'pixel_cy', 'pixel_w', 'pixel_h']
        self.obs_df = pd.DataFrame(self.obs_list, columns=columns)
        self.obs_df['dis'] = self.obs_df['x']**2 + self.obs_df['y']**2
    
    def publish_new_refline(self):
        if len(self.obs_list) == 0:
            print("No obs in the scene")
            self.planning = False
        else:
            print("Obs in the scene")
            data_cone = self.obs_df.sort_values(by='dis', ascending=True)
            ## 以data_cone的x坐标和y坐标画图
            # plt.plot(data_cone["x"], data_cone["y"], "r", label="Cone coords")
            ## 将local_x和local_y转换为utm坐标
            data_cone["utm_coords"] = data_cone.apply(lambda row: self.localxy2utm([row['x'], row['y'], 0, 0], self.state), axis=1)
            data_cone["utm_x"] = data_cone["utm_coords"].apply(lambda x: x[0])
            data_cone["utm_y"] = data_cone["utm_coords"].apply(lambda x: x[1])
            print("utm_x:", data_cone["utm_x"])
            print("utm_y:", data_cone["utm_y"])
            data_cone["utm_yaw"] = data_cone["utm_coords"].apply(lambda x: x[2])
            # 画出utm坐标系下的cone
            plt.scatter(data_cone["utm_x"], data_cone["utm_y"], c="b", linewidth=10, label="Cone utm coords")       
            data_cone = [data_cone.iloc[0]['x'], data_cone.iloc[0]['y'], 
                         2.0, 2.0]
            if self.planning == False:
                self.GenerateLaneBorrow(data_cone)
                self.planning = True
        return self.cx, self.cy, self.cyaw, self.ck, self.sp
    
    def localxy2utm(self, point, state):
        x = point[0]
        y = point[1]
        local_yaw = point[2]
        heading_rad = math.pi/2 - state.yaw
        dx = x * math.cos(heading_rad) + y * math.sin(heading_rad)
        dy = -x * math.sin(heading_rad) + y * math.cos(heading_rad)
        utm_e = state.x + dx
        utm_n = state.y + dy
        # 将局部航向转换到全局航向
        # global_yaw = local_yaw + (math.pi/2 - state.yaw)
        global_yaw = local_yaw - (math.pi/2 - state.yaw)
        # 将角度归一化到(-π, π)
        # global_yaw = (global_yaw + math.pi) % (2 * math.pi) - math.pi
        return [utm_e, utm_n, global_yaw, point[3]]
    
    def utm2localxy(self, state, point_x, point_y):
        det_x = point_x - state.x
        det_y = point_y - state.y
        distance = math.sqrt(det_x ** 2 + det_y ** 2)
        angle_line = math.atan2(det_y, det_x)
        angle = (angle_line - state.yaw + math.pi / 2)
        new_x = distance * math.cos(angle)
        new_y = distance * math.sin(angle)
        return new_x, new_y, angle
   
    def GenerateLaneBorrow(self, obs): 
        print("obs:", obs)
        """lane borrow"""
        # [x, y, width, length]
        if len(obs) == 0:
            return
        index1, index2 = -1, -1
        for i in range(len(self.ref_line)):
            if index1 == -1 and self.ref_line[i][1] >= obs[1] - obs[3] / 2 - 10:
                index1 = i
            if index2 == -1 and self.ref_line[i][1] >= obs[1] + obs[3] / 2 + 10:
                index2 = i
                break
        if index1 == -1 or index2 == -1:
            return
        point0 = [self.ref_line[index1][0], self.ref_line[index1][1]]
        point1 = [obs[0] - obs[2], obs[1] - obs[3] / 2]
        point2 = [obs[0] - obs[2], obs[1] + obs[3] / 2]
        point3 = [self.ref_line[index2][0], self.ref_line[index2][1]]

        b_line1 = self.generate_bezier(point0, point1)
        b_line2 = self.generate_bezier(point1, point2)
        b_line3 = self.generate_bezier(point2, point3)

        # montage
        b_line_all = []
        b_line_all.extend(b_line1)
        b_line_all.extend(b_line2)
        b_line_all.extend(b_line3)

        for one_s_point in b_line_all:
            new_one_point = self.localxy2utm(one_s_point, self.state)
            self.new_line.append([new_one_point[0], new_one_point[1],
                                     new_one_point[2], one_s_point[3]])
        history_x = [point[0] for point in self.new_line]
        history_y = [point[1] for point in self.new_line]
        history_yaw = [point[2] for point in self.new_line]

        self.cx = self.cx[:index1] + history_x + self.cx[index2:]
        self.cy = self.cy[:index1] + history_y + self.cy[index2:]
        self.cyaw = self.cyaw[:index1] + history_yaw + self.cyaw[index2:]
        self.sp = self.calc_speed_profile(self.target_speed)
        # df = pd.DataFrame(self.all_line_temp, columns=['x', 'y', 'yaw', 'v', 'length'])
        # df.to_csv('all_line.csv', index=False)
    
    def calc_speed_profile(self, target_speed):
        speed_profile = [target_speed] * len(self.cx)
        direction = 1.0  # forward
        # Set stop point
        for i in range(len(self.cx) - 1):
            dx = self.cx[i + 1] - self.cx[i]
            dy = self.cy[i + 1] - self.cy[i]
            move_direction = math.atan2(dy, dx)
            if dx != 0.0 and dy != 0.0:
                dangle = abs(pi_2_pi(move_direction - self.cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0
            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed
        speed_profile[-1] = 0.0
        return speed_profile

    def generate_bezier(self, point0, point1):
        b_line = []
        first_control_point_para_ = 0.3
        second_control_point_para_ = 0.4
        y = point1[1] - point0[1]
        CtrlPointX = [0, 0, 0, 0]
        CtrlPointY = [0, 0, 0, 0]
        CtrlPointX[0] = point0[0]
        CtrlPointY[0] = point0[1]
        CtrlPointX[3] = point1[0]
        CtrlPointY[3] = point1[1]
        CtrlPointX[1] = CtrlPointX[0]
        CtrlPointY[1] = y * first_control_point_para_ + point0[1]
        CtrlPointX[2] = CtrlPointX[3]
        CtrlPointY[2] = y * second_control_point_para_ + point0[1]
        Pos = round(y / 1.0)
        for i in range(Pos, 0, -1):
            tempx = self.bezier3func(i / Pos, CtrlPointX)
            tempy = self.bezier3func(i / Pos, CtrlPointY)
            angle, curvature = self.cal_angle_curvature(i / Pos, CtrlPointX, CtrlPointY)
            if angle > 2 * math.pi:
                angle = angle - 2 * math.pi
            elif angle < 0:
                angle = angle + 2 * math.pi
            b_point = [tempx, tempy, angle, curvature]
            b_line.append(b_point)
        return b_line
    
    def bezier3func(self, _t, controlP):
        part0 = controlP[0] * _t * _t * _t
        part1 = 3 * controlP[1] * _t * _t * (1 - _t)
        part2 = 3 * controlP[2] * _t * (1 - _t) * (1 - _t)
        part3 = controlP[3] * (1 - _t) * (1 - _t) * (1 - _t)
        return part0 + part1 + part2 + part3
    
    def cal_angle_curvature(self, _t, controlP_x, controlP_y):
        _dx_1 = 3 * controlP_x[0] * _t * _t
        _dx_2 = 3 * controlP_x[1] * (_t * 2 - 3 * _t * _t)
        _dx_3 = 3 * controlP_x[2] * (1 - 4 * _t + 3 * _t * _t)
        _dx_4 = -3 * controlP_x[3] * (1 - _t) * (1 - _t)
        _dy_1 = 3 * controlP_y[0] * _t * _t
        _dy_2 = 3 * controlP_y[1] * (_t * 2 - 3 * _t * _t)
        _dy_3 = 3 * controlP_y[2] * (1 - 4 * _t + 3 * _t * _t)
        _dy_4 = -3 * controlP_y[3] * (1 - _t) * (1 - _t)
        
        dx = _dx_1 + _dx_2 + _dx_3 + _dx_4
        dy = _dy_1 + _dy_2 + _dy_3 + _dy_4
        
        _ddx_1 = 6 * controlP_x[0] * _t
        _ddx_2 = 6 * controlP_x[1] * (2 - 6 * _t)
        _ddx_3 = 6 * controlP_x[2] * (3 - 8 * _t)
        _ddx_4 = -6 * controlP_x[3] * (1 - _t)
        _ddy_1 = 6 * controlP_y[0] * _t
        _ddy_2 = 6 * controlP_y[1] * (2 - 6 * _t)
        _ddy_3 = 6 * controlP_y[2] * (3 - 8 * _t)
        _ddy_4 = -6 * controlP_y[3] * (1 - _t)
        
        ddx = _ddx_1 + _ddx_2 + _ddx_3 + _ddx_4
        ddy = _ddy_1 + _ddy_2 + _ddy_3 + _ddy_4
        
        angle = math.atan2(-dy, -dx)
        curvature = abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
        
        return angle, curvature