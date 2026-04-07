#include <chrono>
#include <memory>
#include <string>

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace ga_ocp_rviz_plugins {

class PlanningPanel : public rviz_common::Panel {
public:
  explicit PlanningPanel(QWidget* parent = nullptr) : rviz_common::Panel(parent) {
    auto* root_layout = new QVBoxLayout();
    status_label_ = new QLabel("Status: idle");
    detail_label_ = new QLabel("Drag the interactive marker to set the goal pose.");

    auto* button_layout = new QHBoxLayout();
    auto* plan_button = new QPushButton("Plan");
    auto* execute_button = new QPushButton("Execute");
    auto* plan_execute_button = new QPushButton("Plan&Execute");
    auto* clear_button = new QPushButton("Clear");

    button_layout->addWidget(plan_button);
    button_layout->addWidget(execute_button);
    button_layout->addWidget(plan_execute_button);
    button_layout->addWidget(clear_button);

    root_layout->addWidget(status_label_);
    root_layout->addWidget(detail_label_);
    root_layout->addLayout(button_layout);
    setLayout(root_layout);

    node_ = std::make_shared<rclcpp::Node>("ga_ocp_planning_panel");
    status_sub_ = node_->create_subscription<std_msgs::msg::String>(
        "/planning_status", rclcpp::QoS(1).reliable().transient_local(),
        [this](const std_msgs::msg::String::SharedPtr msg) {
          status_label_->setText(QString::fromStdString(msg->data));
        });

    plan_client_ = node_->create_client<std_srvs::srv::Trigger>("/plan_trajectory_srv");
    execute_client_ = node_->create_client<std_srvs::srv::Trigger>("/execute_planned_trajectory_srv");
    clear_client_ = node_->create_client<std_srvs::srv::Trigger>("/clear_planned_trajectory_srv");

    connect(plan_button, &QPushButton::clicked, this, [this]() {
      requestService(plan_client_, "Plan requested.");
    });
    connect(execute_button, &QPushButton::clicked, this, [this]() {
      requestService(execute_client_, "Execute requested.");
    });
    connect(clear_button, &QPushButton::clicked, this, [this]() {
      requestService(clear_client_, "Clear requested.");
    });
    connect(plan_execute_button, &QPushButton::clicked, this, [this]() {
      detail_label_->setText("Plan&Execute requested.");
      if (!plan_client_->wait_for_service(std::chrono::milliseconds(200))) {
        detail_label_->setText("Plan service unavailable.");
        return;
      }
      auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
      pending_plan_execute_ = true;
      pending_plan_future_ = plan_client_->async_send_request(request).future.share();
    });

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, [this]() { spinOnce(); });
    timer_->start(50);
  }

private:
  void requestService(const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr& client,
                      const std::string& pending_text) {
    detail_label_->setText(QString::fromStdString(pending_text));
    if (!client->wait_for_service(std::chrono::milliseconds(200))) {
      detail_label_->setText("Service unavailable.");
      return;
    }
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    pending_future_ = client->async_send_request(request).future.share();
  }

  void spinOnce() {
    if (rclcpp::ok()) {
      rclcpp::spin_some(node_);
    }

    if (pending_future_.valid() &&
        pending_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      const auto response = pending_future_.get();
      detail_label_->setText(QString::fromStdString(response->message));
      pending_future_ = {};
    }

    if (pending_plan_execute_ && pending_plan_future_.valid() &&
        pending_plan_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      const auto response = pending_plan_future_.get();
      detail_label_->setText(QString::fromStdString(response->message));
      pending_plan_future_ = {};
      pending_plan_execute_ = false;
      if (response->success) {
        requestService(execute_client_, "Execute requested after planning.");
      }
    }
  }

  QLabel* status_label_{nullptr};
  QLabel* detail_label_{nullptr};
  QTimer* timer_{nullptr};

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr plan_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr execute_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clear_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture pending_future_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture pending_plan_future_;
  bool pending_plan_execute_{false};
};

}  // namespace ga_ocp_rviz_plugins

PLUGINLIB_EXPORT_CLASS(ga_ocp_rviz_plugins::PlanningPanel, rviz_common::Panel)
