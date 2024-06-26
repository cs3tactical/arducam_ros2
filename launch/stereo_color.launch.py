from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    namespace = LaunchConfiguration('namespace')
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='arducam'
    )
    
    arducam_device = LaunchConfiguration('arducam_device')
    arducam_device_arg = DeclareLaunchArgument(
        'arducam_device',
        default_value='0'
    )

    width = LaunchConfiguration('width')
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='3840'
    )

    height = LaunchConfiguration('height')
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='1200'
    )

    frame_id = LaunchConfiguration('frame_id')
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='stereo_cam'
    )

    fps = LaunchConfiguration('fps')
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='37'
    )

    pixelformat = LaunchConfiguration('pixelformat')
    pixelformat_arg = DeclareLaunchArgument(
        'pixelformat',
        default_value='Y16'
    )

    is_grey = LaunchConfiguration('is_grey')
    is_grey_arg = DeclareLaunchArgument(
        'is_grey',
        default_value='False'
    )

    out_grey = LaunchConfiguration('out_grey')
    out_grey_arg = DeclareLaunchArgument(
        'out_grey',
        default_value='False'
    )

    flip = LaunchConfiguration('flip')
    flip_arg = DeclareLaunchArgument(
        'flip',
        default_value='False'
    )

    resize_factor = LaunchConfiguration('resize_factor')
    resize_factor_arg = DeclareLaunchArgument(
        'resize_factor',
        default_value='3'
    )

    exposure = LaunchConfiguration('exposure')
    exposure_arg = DeclareLaunchArgument(
        'exposure',
        default_value='200'
    )

    l_file_name = 'left_cam_info.yaml'
    package_share_directory = get_package_share_directory('arducam_ros2')
    l_file_path = os.path.join(package_share_directory, l_file_name)
    left_cam_info_file = LaunchConfiguration('left_cam_info_file')
    left_cam_info_file_arg = DeclareLaunchArgument(
        'left_cam_info_file',
        default_value=l_file_path
    )

    r_file_name = 'right_cam_info.yaml'
    r_file_path = os.path.join(package_share_directory, r_file_name)
    right_cam_info_file = LaunchConfiguration('right_cam_info_file')
    right_cam_info_file_arg = DeclareLaunchArgument(
        'right_cam_info_file',
        default_value=r_file_path
    )

    stereo_node = Node(
        package='arducam_ros2',
        executable='arducam_stereo',
        name='arducam_stereo',
        namespace=namespace,
        output='screen',
        parameters=[
                        {'device': arducam_device}, 
                        {'frame_id': frame_id},
                        {'width': width},
                        {'height': height},
                        {'pixelformat': pixelformat},
                        {'left_cam_info_file': left_cam_info_file},
                        {'right_cam_info_file': right_cam_info_file},
                        {'is_grey': is_grey},
                        {'exposure': exposure},
                        {'out_grey': out_grey},
                        {'resize_factor': resize_factor},
                        {'flip': flip},
                        {'fps': fps}
                    ],
    )

    ld = LaunchDescription()

    ld.add_action(namespace_arg)
    ld.add_action(arducam_device_arg)
    ld.add_action(width_arg)
    ld.add_action(height_arg)
    ld.add_action(frame_id_arg)
    ld.add_action(pixelformat_arg)
    ld.add_action(is_grey_arg)
    ld.add_action(exposure_arg)
    ld.add_action(left_cam_info_file_arg)
    ld.add_action(right_cam_info_file_arg)
    ld.add_action(flip_arg)
    ld.add_action(resize_factor_arg)
    ld.add_action(out_grey_arg)
    ld.add_action(fps_arg)
    ld.add_action(stereo_node)

    return ld
