from setuptools import setup

package_name = 'follow_traj_wd'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hcx',
    maintainer_email='hcx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test' : ['pytest'],
    },
    entry_points={
        'console_scripts': [
         'can_node             = follow_traj_wd.can_node:main',
         'adjust_trajectory    = follow_traj_wd.adjust_trajectory_node:main',
         'follow_node          = follow_traj_wd.follow_node:main',
         'mpc_follow_node      = follow_traj_wd.mpc_node_v2:main',
         'hybrid_follow_node   = follow_traj_wd.hybrid_follow_node:main',
         'publish_trajectory   = follow_traj_wd.publish_trajectory_node:main',
         'visualization_node   = follow_traj_wd.visualization_node:main',
        ],
    },
)
