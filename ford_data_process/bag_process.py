#!/usr/bin/env python3

import argparse
import os
import rospy
import rosbag
import pandas as pd
from sensor_msgs.msg import Imu, TimeReference
from geometry_msgs.msg import PoseStamped, Vector3Stamped
import csv


def bag_to_csv(bag_file):
    gps_csv = 'gps_time.csv'
    imu_csv = 'imu_topic.csv'
    bag = rosbag.Bag(bag_file)

    gps_time_data = []
    imu_data = []

    for topic, msg, t in bag.read_messages(topics=['/gps_time', '/imu']):
        if topic == '/gps_time':
            # GPS 시간 데이터를 csv에 기록
            gps_time_data.append([t.to_sec(), msg.data])  # msg.data를 실제 메시지 타입에 맞게 수정 필요

        elif topic == '/imu':
            # IMU 데이터를 csv에 기록
            imu_data.append([t.to_sec(),
                             msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                             msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    # GPS 시간 데이터를 csv 파일로 저장
    gps_time_df = pd.DataFrame(gps_time_data, columns=['Time', 'GPS_Time'])
    gps_time_df.to_csv(gps_csv, index=False)

    # IMU 데이터를 csv 파일로 저장
    imu_df = pd.DataFrame(imu_data, columns=['Time', 'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Orientation_W',
                                             'Angular_Velocity_X', 'Angular_Velocity_Y', 'Angular_Velocity_Z',
                                             'Linear_Acceleration_X', 'Linear_Acceleration_Y', 'Linear_Acceleration_Z'])
    imu_df.to_csv(imu_csv, index=False)

    bag.close()


if __name__ == '__main__':

    rospy.init_node('bag_to_csv')

    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_file', type=str, default='none')
    args = parser.parse_intermixed_args()

    # ROS bag 파일 경로
    root_dir = '/ws/data/Ford-AV'
    if args.bag_file == 'none':
        pass
    else:
        bag_file = os.path.join(root_dir, args.bag_file)
        bag_to_csv(bag_file)
