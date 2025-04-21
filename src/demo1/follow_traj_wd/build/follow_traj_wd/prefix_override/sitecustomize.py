import sys
if sys.prefix == '/home/renth/anaconda3/envs/demo1':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/renth/hezi_ros2/src/demo1/follow_traj_wd/install/follow_traj_wd'
