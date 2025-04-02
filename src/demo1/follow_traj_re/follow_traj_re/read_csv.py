import csv
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
