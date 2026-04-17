import os
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_collision_obstacles(path: str) -> tuple[list[float], list[float]]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    params = config.get('crocoddyl_collision_planner_node', {}).get('ros__parameters', {})
    radii = list(params.get('obstacle_radii', [0.15, 0.12, 0.18]))
    centers = list(params.get('obstacle_centers', [0.3, 0.5, 0.9, -0.2, 0.4, 1.0, 0.5, -0.3, 0.8]))
    return radii, centers


def _generate_mujoco_collision_scene(mujoco_share: str, planner_config: str) -> str:
    radii, centers = _load_collision_obstacles(planner_config)
    if len(centers) != len(radii) * 3:
      raise ValueError(
          f"Invalid obstacle_centers length in {planner_config}: got {len(centers)} expected {len(radii) * 3}"
      )

    material_cycle = ['obs_red', 'obs_green', 'obs_blue', 'obs_yellow', 'obs_cyan', 'obs_magenta']
    material_defs = [
        '    <material name="obs_red" rgba="1.0 0.2 0.2 0.45"/>',
        '    <material name="obs_green" rgba="0.2 1.0 0.2 0.45"/>',
        '    <material name="obs_blue" rgba="0.2 0.4 1.0 0.45"/>',
        '    <material name="obs_yellow" rgba="1.0 0.9 0.2 0.45"/>',
        '    <material name="obs_cyan" rgba="0.2 0.9 0.9 0.45"/>',
        '    <material name="obs_magenta" rgba="0.9 0.2 0.9 0.45"/>',
    ]

    obstacle_lines = []
    for idx, radius in enumerate(radii):
        cx, cy, cz = centers[3 * idx: 3 * idx + 3]
        material = material_cycle[idx % len(material_cycle)]
        obstacle_lines.extend([
            f'    <body name="obstacle_{idx + 1}" pos="{cx} {cy} {cz}">',
            (
                f'      <geom name="obstacle_{idx + 1}_geom" type="sphere" size="{radius}" '
                f'material="{material}" contype="1" conaffinity="1"/>'
            ),
            '    </body>',
            '',
        ])

    scene_text = "\n".join([
        '<mujoco model="ur10_collision_scene_generated">',
        f'  <include file="{os.path.join(mujoco_share, "robot-assets", "ur10", "mjcf", "ur10.xml")}"/>',
        '',
        '  <statistic center="0.4 0 0.4" extent="1"/>',
        '',
        '  <visual>',
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="120" elevation="-20"/>',
        '  </visual>',
        '',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        (
            '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" '
            'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>'
        ),
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>',
        '',
        *material_defs,
        '  </asset>',
        '',
        '  <worldbody>',
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
        '',
        *obstacle_lines,
        '  </worldbody>',
        '</mujoco>',
        '',
    ])

    scene_path = os.path.join(tempfile.gettempdir(), 'ga_ocp_ur10_collision_scene_generated.xml')
    with open(scene_path, 'w', encoding='utf-8') as f:
        f.write(scene_text)
    return scene_path


def generate_launch_description() -> LaunchDescription:
    ros2_share = get_package_share_directory('ga_ocp_ros2')
    core_share = get_package_share_directory('ga_ocp_core')
    urdf_path = f"{core_share}/robot-assets/ur10/urdf/ur10.urdf"
    rviz_config = f"{ros2_share}/rviz/ga_ocp_collision_validation.rviz"
    planner_config = f"{ros2_share}/config/collision_planner.yaml"
    marker_config = f"{ros2_share}/config/target_interactive_marker.yaml"
    collision_scene = _generate_mujoco_collision_scene(core_share, planner_config)

    use_rviz = LaunchConfiguration('rviz')

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
        executable='crocoddyl_collision_planner_node',
        name='crocoddyl_collision_planner_node',
        output='screen',
        parameters=[planner_config],
    )

    mujoco_executor_node = Node(
        package='ga_ocp_ros2',
        executable='joint_command_executor.py',
        name='mujoco_joint_executor_node',
        output='screen',
        parameters=[{'robot': 'ur', 'xml_file': collision_scene}],
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
        robot_state_publisher_node,
        target_marker_node,
        planner_node,
        mujoco_executor_node,
        rviz_node,
    ])
