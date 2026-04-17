from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    ros2_share = get_package_share_directory('ga_ocp_ros2')
    config = f"{ros2_share}/config/closed_loop_mpc_ur.yaml"

    backend = LaunchConfiguration('backend')
    solve_budget_ms = LaunchConfiguration('solve_budget_ms')
    duration_s = LaunchConfiguration('duration_s')
    mass_scale = LaunchConfiguration('mass_scale')
    plant_payload_mass = LaunchConfiguration('plant_payload_mass')
    controller_payload_mass = LaunchConfiguration('controller_payload_mass')
    model_payload = LaunchConfiguration('model_payload')
    output_prefix = LaunchConfiguration('output_prefix')

    closed_loop_node = Node(
        package='ga_ocp_ros2',
        executable='closed_loop_mpc_node',
        name='closed_loop_mpc_node',
        output='screen',
        parameters=[
            config,
            {
                'backend': backend,
                'solve_budget_ms': ParameterValue(solve_budget_ms, value_type=float),
                'experiment_duration_s': ParameterValue(duration_s, value_type=float),
                'plant_mass_scale': ParameterValue(mass_scale, value_type=float),
                'plant_payload_mass': ParameterValue(plant_payload_mass, value_type=float),
                'controller_payload_mass': ParameterValue(controller_payload_mass, value_type=float),
                'model_payload': ParameterValue(model_payload, value_type=bool),
                'output_prefix': output_prefix,
            },
        ],
    )

    mujoco_executor_node = Node(
        package='ga_ocp_ros2',
        executable='joint_command_executor.py',
        name='mujoco_joint_executor_node',
        output='screen',
        parameters=[
            {
                'robot': 'ur',
                'mass_scale': ParameterValue(mass_scale, value_type=float),
                'payload_mass': ParameterValue(plant_payload_mass, value_type=float),
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='tetrapga'),
        DeclareLaunchArgument('solve_budget_ms', default_value='10.0'),
        DeclareLaunchArgument('duration_s', default_value='20.0'),
        DeclareLaunchArgument('mass_scale', default_value='1.0'),
        DeclareLaunchArgument('plant_payload_mass', default_value='0.0'),
        DeclareLaunchArgument('controller_payload_mass', default_value='0.0'),
        DeclareLaunchArgument('model_payload', default_value='false'),
        DeclareLaunchArgument('output_prefix', default_value=''),
        RegisterEventHandler(
            OnProcessExit(
                target_action=closed_loop_node,
                on_exit=[EmitEvent(event=Shutdown(reason='closed-loop mpc node finished'))],
            )
        ),
        closed_loop_node,
        mujoco_executor_node,
    ])
