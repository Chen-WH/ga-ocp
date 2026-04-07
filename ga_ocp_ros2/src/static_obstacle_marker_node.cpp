#include <chrono>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class StaticObstacleMarkerNode : public rclcpp::Node {
public:
  StaticObstacleMarkerNode() : Node("static_obstacle_marker_node") {
    marker_frame_id_ = this->declare_parameter<std::string>("marker_frame_id", "base_link");
    obstacle_radii_ = this->declare_parameter<std::vector<double>>("obstacle_radii", {0.15, 0.12, 0.18});
    obstacle_centers_ = this->declare_parameter<std::vector<double>>(
        "obstacle_centers", {0.3, 0.5, 0.9, -0.2, 0.4, 1.0, 0.5, -0.3, 0.8});

    if (obstacle_centers_.size() != obstacle_radii_.size() * 3) {
      RCLCPP_WARN(
          this->get_logger(),
          "Invalid obstacle_centers length (%zu). Expected %zu. Obstacle markers will not be published.",
          obstacle_centers_.size(), obstacle_radii_.size() * 3);
    }

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/ga_ocp/collision_scene", rclcpp::QoS(1).reliable().transient_local());
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000), std::bind(&StaticObstacleMarkerNode::publishMarkers, this));

    publishMarkers();
  }

private:
  void publishMarkers() {
    visualization_msgs::msg::MarkerArray array;
    const auto stamp = this->now();

    if (obstacle_centers_.size() != obstacle_radii_.size() * 3) {
      marker_pub_->publish(array);
      return;
    }

    for (size_t i = 0; i < obstacle_radii_.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = marker_frame_id_;
      marker.header.stamp = stamp;
      marker.ns = "obstacles";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = obstacle_centers_[3 * i + 0];
      marker.pose.position.y = obstacle_centers_[3 * i + 1];
      marker.pose.position.z = obstacle_centers_[3 * i + 2];
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * obstacle_radii_[i];
      marker.scale.y = 2.0 * obstacle_radii_[i];
      marker.scale.z = 2.0 * obstacle_radii_[i];
      marker.color.a = 0.45f;
      marker.color.r = (i == 0) ? 1.0f : 0.2f;
      marker.color.g = (i == 1) ? 1.0f : 0.2f;
      marker.color.b = (i == 2) ? 1.0f : 0.4f;
      array.markers.push_back(marker);
    }

    marker_pub_->publish(array);
  }

  std::string marker_frame_id_;
  std::vector<double> obstacle_radii_;
  std::vector<double> obstacle_centers_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StaticObstacleMarkerNode>());
  rclcpp::shutdown();
  return 0;
}
