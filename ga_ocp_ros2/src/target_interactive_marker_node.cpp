#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>

class EETargetInteractiveMarkerNode : public rclcpp::Node {
public:
  EETargetInteractiveMarkerNode()
      : Node("target_interactive_marker_node"),
        server_(std::make_shared<interactive_markers::InteractiveMarkerServer>(
            "target_interactive_marker", this)) {
    target_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/target_ee_pose", 10);

    marker_frame_id_ = this->declare_parameter<std::string>("marker_frame_id", "base_link");
    marker_scale_ = this->declare_parameter<double>("marker_scale", 0.25);

    target_pose_.position.x = this->declare_parameter<double>("target_x", 0.0);
    target_pose_.position.y = this->declare_parameter<double>("target_y", 0.2561);
    target_pose_.position.z = this->declare_parameter<double>("target_z", 1.4273);
    target_pose_.orientation.x = this->declare_parameter<double>("target_qx", -0.7071);
    target_pose_.orientation.y = this->declare_parameter<double>("target_qy", 0.0);
    target_pose_.orientation.z = this->declare_parameter<double>("target_qz", 0.0);
    target_pose_.orientation.w = this->declare_parameter<double>("target_qw", 0.7071);

    publishInteractiveMarker();
    publishCurrentGoal();

    RCLCPP_INFO(this->get_logger(),
                "Interactive marker ready. Drag target to set goal pose, then use the RViz Planning panel.");
  }

private:
  void publishInteractiveMarker() {
    visualization_msgs::msg::InteractiveMarker int_marker;
    int_marker.header.frame_id = marker_frame_id_;
    int_marker.name = "ee_target";
    int_marker.description = "EE Target";
    int_marker.scale = marker_scale_;
    int_marker.pose = target_pose_;

    visualization_msgs::msg::Marker box;
    box.type = visualization_msgs::msg::Marker::CUBE;
    box.scale.x = 0.06;
    box.scale.y = 0.06;
    box.scale.z = 0.06;
    box.color.r = 0.2f;
    box.color.g = 0.8f;
    box.color.b = 0.2f;
    box.color.a = 0.9f;

    visualization_msgs::msg::InteractiveMarkerControl box_control;
    box_control.always_visible = true;
    box_control.markers.push_back(box);
    int_marker.controls.push_back(box_control);

    add6DofControls(int_marker);

    server_->insert(int_marker, std::bind(&EETargetInteractiveMarkerNode::onMarkerFeedback, this, std::placeholders::_1));
    server_->applyChanges();
  }

  void addAxisControl(visualization_msgs::msg::InteractiveMarker& marker, const std::string& name,
                      double ox, double oy, double oz,
                      uint8_t interaction_mode) {
    visualization_msgs::msg::InteractiveMarkerControl control;
    control.name = name;
    control.orientation.w = 1.0;
    control.orientation.x = ox;
    control.orientation.y = oy;
    control.orientation.z = oz;
    control.interaction_mode = interaction_mode;
    marker.controls.push_back(control);
  }

  void add6DofControls(visualization_msgs::msg::InteractiveMarker& marker) {
    addAxisControl(marker, "rotate_x", 1.0, 0.0, 0.0,
                   visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS);
    addAxisControl(marker, "move_x", 1.0, 0.0, 0.0,
                   visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS);

    addAxisControl(marker, "rotate_y", 0.0, 0.0, 1.0,
                   visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS);
    addAxisControl(marker, "move_y", 0.0, 0.0, 1.0,
                   visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS);

    addAxisControl(marker, "rotate_z", 0.0, 1.0, 0.0,
                   visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS);
    addAxisControl(marker, "move_z", 0.0, 1.0, 0.0,
                   visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS);
  }

  void onMarkerFeedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
    if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
      target_pose_ = feedback->pose;
      publishCurrentGoal();
    }
  }

  void publishCurrentGoal() {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = marker_frame_id_;
    msg.pose = target_pose_;
    target_pub_->publish(msg);
  }

private:
  std::string marker_frame_id_;
  double marker_scale_;
  geometry_msgs::msg::Pose target_pose_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pub_;
  std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EETargetInteractiveMarkerNode>());
  rclcpp::shutdown();
  return 0;
}
