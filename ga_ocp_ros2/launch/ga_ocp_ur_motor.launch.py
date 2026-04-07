from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_launch_description() -> LaunchDescription:
    ros2_share = get_package_share_directory('ga_ocp_ros2')
    core_share = get_package_share_directory('ga_ocp_core')
    urdf_path = f"{core_share}/description/urdf/ur10.urdf"
    rviz_config = f"{ros2_share}/rviz/ga_ocp_motor_validation.rviz"
    planner_config = f"{ros2_share}/config/motor_planner.yaml"
    marker_config = f"{ros2_share}/config/target_interactive_marker.yaml"

    use_rviz = LaunchConfiguration('rviz')
    robot_ip = LaunchConfiguration('robot_ip')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': _read_text(urdf_path)}],
    )

    target_marker_node = Node(
        package='ga_ocp_ros2',
        executable='target_interactive_marker_node',
        name='target_interactive_marker_node',
        output='screen',
        parameters=[marker_config],
    )

    planner_node = Node(
        package='ga_ocp_ros2',
        executable='crocoddyl_motor_planner_node',
        name='crocoddyl_motor_planner_node',
        output='screen',
        parameters=[planner_config],
    )

    ur_executor_node = Node(
        package='ur-wrapper',
        executable='joint_executor.py',
        name='joint_executor_node',
        output='screen',
        parameters=[{'robot_ip': robot_ip}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.125.6'),
        robot_state_publisher_node,
        target_marker_node,
        planner_node,
        ur_executor_node,
        rviz_node,
    ])
