# ga-ocp

This repository is organized as a multi-package ROS 2 workspace repository.

## Packages

- `ga_ocp_core`: Geometric-algebra optimal control core, including Crocoddyl integration, tests, benchmarks, and robot descriptions used by the current benchmark suite.
- `ga_ocp_ros2`: ROS 2 application-layer package for planner nodes, launch files, RViz configs, and system integration.

## Layout

```text
ga-ocp/
  ga_ocp_core/
  ga_ocp_ros2/
```

## Build

From a ROS 2 workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --parallel-workers 4 --packages-select ga_ocp_core ga_ocp_ros2 \
  --cmake-args -DGA_OCP_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_PARALLEL_LEVEL=4
```

`ga_ocp_core` expects a standard installed dependency:

- `find_package(TetraPGA CONFIG REQUIRED)`

## Notes

- Benchmarks are kept on the main branch but remain disabled by default.
- When benchmark targets are enabled, CasADi-backed benchmark cases are built automatically.
- `ga_ocp_ros2` contains the ROS 2 nodes, launch files, rviz configs, and panel plugin built on top of `ga_ocp_core`.
