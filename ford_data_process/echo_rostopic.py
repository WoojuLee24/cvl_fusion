import rosbag
import csv
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import os
import argparse

def extract_gps_data(bag_file, csv_file):

    with rosbag.Bag(bag_file, 'r') as bag:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['timestamp', 'latitude', 'longitude', 'altitude', 'position_covariance'])
            writer.writerow(['rosbagTimestamp', 'header', 'seq', 'stamp', 'secs', 'nsecs',
                             'frame_id', 'status', 'status', 'service',
                             'latitude', 'longitude', 'altitude', 'position_covariance', 'position_covariance_type'])

            for topic, msg, t in bag.read_messages(topics=['/gps']):
                # print('msg: ', msg)
                # if isinstance(msg, NavSatFix):
                covariance_as_string = ','.join(map(str, msg.position_covariance))
                covariance_as_string = '['+covariance_as_string+']'
                writer.writerow([t, '', msg.header.seq, '', msg.header.stamp.secs, msg.header.stamp.nsecs,
                                 msg.header.frame_id, '', msg.status.status, msg.status.service,
                                 msg.latitude, msg.longitude, msg.altitude,
                                 covariance_as_string, msg.position_covariance_type])

def extract_pose_ground_truth_data(bag_file, csv_file):

    with rosbag.Bag(bag_file, 'r') as bag:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['timestamp', 'latitude', 'longitude', 'altitude', 'position_covariance'])
            writer.writerow(['rosbagTimestamp', 'header', 'seq', 'stamp', 'secs', 'nsecs',
                             'frame_id', 'pose', 'position', 'x', 'y', 'z',
                             'orientation', 'x', 'y', 'z', 'w'])

            for topic, msg, t in bag.read_messages(topics=['/pose_ground_truth']):
                # print('msg: ', msg)
                # if isinstance(msg, NavSatFix):

                writer.writerow([t, '', msg.header.seq, '', msg.header.stamp.secs, msg.header.stamp.nsecs,
                                 msg.header.frame_id, '', '', msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                 '', msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])


if __name__ == '__main__':
    log_list = ['2017-07-24-V2-Log1',
                '2017-07-24-V2-Log2',
                '2017-07-24-V2-Log3',
                '2017-07-24-V2-Log4',
                '2017-07-24-V2-Log5',
                '2017-07-24-V2-Log6',
                '2017-08-04-V2-Log1',
                '2017-08-04-V2-Log2',
                '2017-08-04-V2-Log3',
                '2017-08-04-V2-Log4',
                '2017-08-04-V2-Log5',
                '2017-08-04-V2-Log6',
                '2017-10-26-V2-Log1',
                '2017-10-26-V2-Log2',
                '2017-10-26-V2-Log3',
                '2017-10-26-V2-Log4',
                '2017-10-26-V2-Log5',
                '2017-10-26-V2-Log6',
                ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='gps')
    args = parser.parse_intermixed_args()

    for log in log_list:
        bag_file_path = f'/ws/data/Ford_AV/bag/{log}.bag'
        csv_folder = f'/ws/data/Ford_AV/{log}'
        csv_file_path = f'/ws/data/Ford_AV/{log}/info_files/{args.topic}.csv'

        if not os.path.exists(bag_file_path):
            print(f"Error: The file {bag_file_path} does not exist.")
            continue
        else:
            print(f"File {bag_file_path} found.")

        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder, exist_ok=True)
        if os.path.exists(csv_file_path):
            print(f"File {csv_file_path} found.")
            csv_file_path = csv_file_path[:-4] + '_2.csv'

        if args.topic == 'gps':
            extract_gps_data(bag_file_path, csv_file_path)
        elif args.topic == 'pose_ground_truth':
            extract_pose_ground_truth_data(bag_file_path, csv_file_path)
