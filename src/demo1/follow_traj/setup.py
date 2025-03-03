from setuptools import setup

package_name = 'follow_traj'

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
         'mpc_node  = follow_traj.mpc_node:main',
         'simulator_node  = follow_traj.simulator_node:main',
        ],
    },
)
