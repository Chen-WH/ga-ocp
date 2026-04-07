#pragma once

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/core/state-base.hpp>
#include <Eigen/Dense>
#include "TetraPGA/Kinematics.hpp"
#include "TetraPGA/Dynamics.hpp"
#include "TetraPGA/Collision.hpp"

using namespace TetraPGA;

/****** define the Differential Action Model ******/

template <typename Scalar>
class DifferentialActionModelGA;

template <typename Scalar>
struct DifferentialActionDataGA : public crocoddyl::DifferentialActionDataAbstractTpl<Scalar>,
                                   public crocoddyl::DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef crocoddyl::CostDataSumTpl<Scalar> CostDataSum;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  Data<Scalar> ga_data;
  std::shared_ptr<CostDataSum> costs;

  template <typename Model>
  explicit DifferentialActionDataGA(Model* const model);
};

template <typename Scalar>
class DifferentialActionModelGA : public crocoddyl::DifferentialActionModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  friend struct DifferentialActionDataGA<Scalar>;

  DifferentialActionModelGA(std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>> state,
                    const Model<Scalar>& ga_model,
                    std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> cost_model = nullptr)
      : crocoddyl::DifferentialActionModelAbstractTpl<Scalar>(state, ga_model.dof_a), // nu = dof_a (假设全驱动)
        ga_model_(ga_model),
        costs_(cost_model) {
    // 检查 State 维度与 GA 模型是否匹配
    if (state->get_nq() != static_cast<std::size_t>(ga_model.dof_a) || 
        state->get_nv() != static_cast<std::size_t>(ga_model.dof_a)) {
    }
  }

  virtual ~DifferentialActionModelGA() {}

  virtual void calc(const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
                    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    // 1. 转换 Data 指针
    DifferentialActionDataGA<Scalar>* d = static_cast<DifferentialActionDataGA<Scalar>*>(data.get());

    // 2. 拆分状态 x = [q; v]
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    // 3. 调用正向动力学
    // forwardDynamics 计算出 acceleration 并存入 ga_data.ddq
    forwardDynamics(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);
    forwardKinematics(ga_model_, d->ga_data, x.head(nq));

    // 4. 将结果赋值给 Crocoddyl 需要的 xout (即 acceleration)
    d->xout = d->ga_data.ddq;

    // 5. 计算 Cost
    if (costs_) {
      costs_->calc(d->costs, x, u);
      d->cost = d->costs->cost;
    } else {
      d->cost = 0;
    }
  }

  virtual void calcDiff(const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
                        const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
	    DifferentialActionDataGA<Scalar>* d = static_cast<DifferentialActionDataGA<Scalar>*>(data.get());
	
	    const std::size_t nq = this->get_state()->get_nq();
	    const std::size_t nv = this->get_state()->get_nv();
	
	    // 1. 调用一阶导数算法
	    // 该函数会填充 ga_data.pddq_pq, ga_data.pddq_pdq, ga_data.pddq_ptau
	    forwardDynamics_fo(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);

    // 2. 填充 Fx (Dynamics Jacobian w.r.t State)
    // Fx = [ da/dq, da/dv ]
    // 注意：Data 中的矩阵是 resize 过的，直接赋值是安全的
    d->Fx.leftCols(nv) = d->ga_data.pddq_pq;
    d->Fx.rightCols(nv) = d->ga_data.pddq_pdq;

    // 3. 填充 Fu (Dynamics Jacobian w.r.t Control)
    d->Fu = d->ga_data.pddq_ptau;

    // 4. 计算 Cost Derivatives 并复制到 data
    if (costs_) {
      costs_->calcDiff(d->costs, x, u);
      // 关键：将 cost 梯度复制到 action data
      d->Lx = d->costs->Lx;
      d->Lu = d->costs->Lu;
      d->Lxx = d->costs->Lxx;
      d->Lxu = d->costs->Lxu;
      d->Luu = d->costs->Luu;
    }
  }

  // ===========================================================================
  // 创建数据结构
  // ===========================================================================
  virtual std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> createData() {
    return std::make_shared<DifferentialActionDataGA<Scalar>>(this);
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsDouble() const {
    // 简化实现：不支持跨标量类型克隆
    throw std::runtime_error("cloneAsDouble not implemented for DifferentialActionModelGA");
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsFloat() const {
    // 简化实现：不支持跨标量类型克隆
    throw std::runtime_error("cloneAsFloat not implemented for DifferentialActionModelGA");
  }

  // ===========================================================================
  // 访问接口
  // ===========================================================================
  const Model<Scalar>& get_ga_model() const { 
      return ga_model_; 
  }

 private:
  Model<Scalar> ga_model_;
  std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> costs_;
};

template <typename Scalar>
template <typename Model>
DifferentialActionDataGA<Scalar>::DifferentialActionDataGA(Model* const model)
    : crocoddyl::DifferentialActionDataAbstractTpl<Scalar>(model),
      ga_data(static_cast<DifferentialActionModelGA<Scalar>*>(model)->get_ga_model())
{
    // 初始化 costs 数据
    auto ga_model_ptr = static_cast<DifferentialActionModelGA<Scalar>*>(model);
    if (ga_model_ptr->costs_) {
        costs = ga_model_ptr->costs_->createData(
            static_cast<DataCollectorAbstract*>(this));
    }
}


/****** define the Acceleration Residual Model for forward dynamics ******/

template <typename Scalar>
class ResidualModelAccelerationGA;

template <typename Scalar>
struct ResidualDataAccelerationGA
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;

  template <typename Model>
  explicit ResidualDataAccelerationGA(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataGA<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataAccelerationGA requires DifferentialActionDataGA as data collector");
    }
    ga_data = &action_data->ga_data;
  }
};

template <typename Scalar>
class ResidualModelAccelerationGA
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelAccelerationGA(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const VectorXs& a_ref)
      : Base(state,
             static_cast<std::size_t>(ga_model.dof_a),
             static_cast<std::size_t>(ga_model.dof_a),
             true,
             true,
             true),
        ga_model_(ga_model),
        a_ref_(a_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataAccelerationGA<Scalar>*>(data.get());
    data->r = d->ga_data->ddq - a_ref_;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataAccelerationGA<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->ga_data->pddq_pq;
    data->Rx.rightCols(nv) = d->ga_data->pddq_pdq;
    data->Ru = d->ga_data->pddq_ptau;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataAccelerationGA<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

 private:
  Model<Scalar> ga_model_;
  VectorXs a_ref_;
};

/****** define the Inverse-Dynamics Differential Action Model ******/

template <typename Scalar>
class DifferentialActionModelGAInv;

template <typename Scalar>
struct DifferentialActionDataGAInv
    : public crocoddyl::DifferentialActionDataAbstractTpl<Scalar>,
      public crocoddyl::DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef crocoddyl::CostDataSumTpl<Scalar> CostDataSum;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  Data<Scalar> ga_data;
  std::shared_ptr<CostDataSum> costs;

  template <typename Model>
  explicit DifferentialActionDataGAInv(Model* const model);
};

template <typename Scalar>
class DifferentialActionModelGAInv
    : public crocoddyl::DifferentialActionModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend struct DifferentialActionDataGAInv<Scalar>;

  DifferentialActionModelGAInv(
      std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>> state,
      const Model<Scalar>& ga_model,
      std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> cost_model = nullptr)
      : crocoddyl::DifferentialActionModelAbstractTpl<Scalar>(
            state, ga_model.dof_a),
        ga_model_(ga_model),
        costs_(cost_model) {}

  virtual ~DifferentialActionModelGAInv() {}

  virtual void calc(
      const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    DifferentialActionDataGAInv<Scalar>* d =
        static_cast<DifferentialActionDataGAInv<Scalar>*>(data.get());

    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    inverseDynamics(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);
    forwardKinematics(ga_model_, d->ga_data, x.head(nq));

    d->xout = u;

    if (costs_) {
      costs_->calc(d->costs, x, u);
      d->cost = d->costs->cost;
    } else {
      d->cost = 0;
    }
  }

  virtual void calcDiff(
      const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    DifferentialActionDataGAInv<Scalar>* d =
        static_cast<DifferentialActionDataGAInv<Scalar>*>(data.get());

    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    inverseDynamics_fo(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);

    d->Fx.setZero();
    d->Fu.setZero();
    d->Fu.leftCols(nv).diagonal().setOnes();

    if (costs_) {
      costs_->calcDiff(d->costs, x, u);
      d->Lx = d->costs->Lx;
      d->Lu = d->costs->Lu;
      d->Lxx = d->costs->Lxx;
      d->Lxu = d->costs->Lxu;
      d->Luu = d->costs->Luu;
    }
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
  createData() {
    return std::make_shared<DifferentialActionDataGAInv<Scalar>>(this);
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase>
  cloneAsDouble() const {
    throw std::runtime_error(
        "cloneAsDouble not implemented for DifferentialActionModelGAInv");
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase>
  cloneAsFloat() const {
    throw std::runtime_error(
        "cloneAsFloat not implemented for DifferentialActionModelGAInv");
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> costs_;
};

template <typename Scalar>
template <typename Model>
DifferentialActionDataGAInv<Scalar>::DifferentialActionDataGAInv(Model* const model)
    : crocoddyl::DifferentialActionDataAbstractTpl<Scalar>(model),
      ga_data(static_cast<DifferentialActionModelGAInv<Scalar>*>(model)->get_ga_model()) {
  auto ga_model_ptr = static_cast<DifferentialActionModelGAInv<Scalar>*>(model);
  this->Fu.setZero();
  this->Fu.leftCols(model->get_state()->get_nv()).diagonal().setOnes();
  if (ga_model_ptr->costs_) {
    costs = ga_model_ptr->costs_->createData(
        static_cast<DataCollectorAbstract*>(this));
  }
}

/****** define the Joint-Effort Residual Model for inverse dynamics ******/

template <typename Scalar>
class ResidualModelJointEffortGAInv;

template <typename Scalar>
struct ResidualDataJointEffortGAInv
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;

  template <typename Model>
  explicit ResidualDataJointEffortGAInv(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataGAInv<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataJointEffortGAInv requires DifferentialActionDataGAInv as data collector");
    }
    ga_data = &action_data->ga_data;
  }
};

template <typename Scalar>
class ResidualModelJointEffortGAInv
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelJointEffortGAInv(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const VectorXs& tau_ref)
      : Base(state,
             static_cast<std::size_t>(ga_model.dof_a),
             static_cast<std::size_t>(ga_model.dof_a),
             true,
             true,
             true),
        ga_model_(ga_model),
        tau_ref_(tau_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataJointEffortGAInv<Scalar>*>(data.get());
    data->r = d->ga_data->tau - tau_ref_;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataJointEffortGAInv<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->ga_data->ptau_pq;
    data->Rx.rightCols(nv) = d->ga_data->ptau_pdq;
    data->Ru = d->ga_data->ptau_pddq;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataJointEffortGAInv<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const VectorXs& get_reference() const { return tau_ref_; }

 private:
  Model<Scalar> ga_model_;
  VectorXs tau_ref_;
};

/****** define the Placement Residual Model ******/

template <typename Scalar>
class ResidualModelFramePlacementGA;

template <typename Scalar, typename Derived>
inline Motor3D<Scalar> align_motor_hemisphere(const Motor3D<Scalar>& reference,
                                              const Eigen::MatrixBase<Derived>& candidate) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 8);
  Motor3D<Scalar> aligned = reference;
  if (aligned.template head<4>().dot(candidate.template head<4>()) < Scalar(0)) {
    aligned = -aligned;
  }
  return aligned;
}

template <typename Scalar>
struct ResidualDataFramePlacementGA
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;
  Line3D<Scalar> r;
  Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J;

  template <typename Model>
  explicit ResidualDataFramePlacementGA(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataGA<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataFramePlacementGA requires DifferentialActionDataGA as data collector");
    }
    ga_data = &action_data->ga_data;
    J.resize(6, ga_data->q.size());
  }
};

template <typename Scalar>
class ResidualModelFramePlacementGA
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelFramePlacementGA(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Motor3D<Scalar>& M_ref)
            : Base(state,
              static_cast<std::size_t>(6),
              static_cast<std::size_t>(state->get_nv()),
              true,
              false,
              false),
        ga_model_(ga_model),
        M_ref_(M_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataFramePlacementGA<Scalar>*>(data.get());

    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = ga_log(ga_mul(ga_rev(align_motor_hemisphere(M_ref_, M_cur)), M_cur));
    data->r = d->r;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataFramePlacementGA<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();

    forwardKinematics(ga_model_, *(d->ga_data), x.head(nq));
    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = ga_log(ga_mul(ga_rev(align_motor_hemisphere(M_ref_, M_cur)), M_cur));

    // Analytic Jacobian of log(M_ref^{-1} M(q))
    analyticJacobian(ga_model_, *(d->ga_data), x.head(nq), d->r);
    d->J = d->ga_data->jac;

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->J;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataFramePlacementGA<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  Motor3D<Scalar> M_ref_;
};

/****** define the Motor Residual Model ******/

template <typename Scalar>
class ResidualModelFrameMotorGA;

template <typename Scalar>
struct ResidualDataFrameMotorGA
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;
  Motor3D<Scalar> r;
  Eigen::Matrix<Scalar, 8, Eigen::Dynamic> J;

  template <typename Model>
  explicit ResidualDataFrameMotorGA(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataGA<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataFrameMotorGA requires DifferentialActionDataGA as data collector");
    }
    ga_data = &action_data->ga_data;
    J.resize(8, ga_data->q.size());
  }
};

template <typename Scalar>
class ResidualModelFrameMotorGA
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelFrameMotorGA(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Motor3D<Scalar>& M_ref)
      : Base(state,
             static_cast<std::size_t>(8),
             static_cast<std::size_t>(state->get_nv()),
             true,
             false,
             false),
        ga_model_(ga_model),
        M_ref_(M_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataFrameMotorGA<Scalar>*>(data.get());

    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = M_cur - align_motor_hemisphere(M_ref_, M_cur);
    data->r = d->r;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataFrameMotorGA<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();

    motorJacobian(ga_model_, *(d->ga_data), x.head(nq));
    d->J = d->ga_data->jacM;

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->J;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataFrameMotorGA<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  Motor3D<Scalar> M_ref_;
};

/****** define the Collision Residual Model ******/

template <typename Scalar>
class ResidualModelCollisionGA;

template <typename Scalar>
struct ResidualDataCollisionGA
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;  // 指向 DifferentialActionDataGA 中的 ga_data
  Environment<Scalar> env;
  EnvironmentData<Scalar> env_data;

  template <typename Model>
  explicit ResidualDataCollisionGA(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        env(static_cast<ResidualModelCollisionGA<Scalar>*>(model)->get_environment()),
        env_data(static_cast<ResidualModelCollisionGA<Scalar>*>(model)->get_ga_model(),
                 static_cast<ResidualModelCollisionGA<Scalar>*>(model)->get_environment()) {
    auto* action_data = dynamic_cast<DifferentialActionDataGA<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataCollisionGA requires DifferentialActionDataGA as data collector");
    }
    ga_data = &(action_data->ga_data);
  }
};

template <typename Scalar>
class ResidualModelCollisionGA
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelCollisionGA(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Environment<Scalar>& env,
      const Scalar d_safe = 0.1)
      : Base(state,
             static_cast<std::size_t>(ga_model.num_collision_ssl * env.num_static_sphere),
             static_cast<std::size_t>(state->get_nv()),
             true,
             false,
             false),
        ga_model_(ga_model),
        env_(env),
        d_safe_(d_safe) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataCollisionGA<Scalar>*>(data.get());
    
    // Compute collision distances for all pairs
    computeDistance(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    // Set residual as (d_safe - distance) for each collision pair
    // Activation model will handle max(0, r) to create barrier
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->r(i) = d_safe_ - d->env_data.distance[i];
    }
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataCollisionGA<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    // Baseline path: recompute witness geometry inside calcDiff.
    computeDistanceJacobian(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    data->Rx.setZero();
    data->Ru.setZero();
    
    // Set Jacobian for each collision pair
    // Negative sign because residual is (d_safe - distance)
    // dr/dq = -d(distance)/dq
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->Rx.row(i).head(nq) = -d->env_data.jac_dist[i];
    }
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataCollisionGA<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }
  const Environment<Scalar>& get_environment() const { return env_; }
  Scalar get_d_safe() const { return d_safe_; }

 private:
  Model<Scalar> ga_model_;
  Environment<Scalar> env_;
  Scalar d_safe_;  // Safety distance threshold
};

/****************Cache version******************************/

template <typename Scalar>
class ResidualModelCollisionCacheGA;

template <typename Scalar>
struct ResidualDataCollisionCacheGA
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;  // 指向 DifferentialActionDataGA 中的 ga_data
  Environment<Scalar> env;
  EnvironmentData<Scalar> env_data;

  template <typename Model>
  explicit ResidualDataCollisionCacheGA(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        env(static_cast<ResidualModelCollisionCacheGA<Scalar>*>(model)->get_environment()),
        env_data(static_cast<ResidualModelCollisionCacheGA<Scalar>*>(model)->get_ga_model(),
                 static_cast<ResidualModelCollisionCacheGA<Scalar>*>(model)->get_environment()) {
    auto* action_data = dynamic_cast<DifferentialActionDataGA<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataCollisionCacheGA requires DifferentialActionDataGA as data collector");
    }
    ga_data = &(action_data->ga_data);
  }
};

template <typename Scalar>
class ResidualModelCollisionCacheGA
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelCollisionCacheGA(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Environment<Scalar>& env,
      const Scalar d_safe = 0.1)
      : Base(state,
             static_cast<std::size_t>(ga_model.num_collision_ssl * env.num_static_sphere),
             static_cast<std::size_t>(state->get_nv()),
             true,
             false,
             false),
        ga_model_(ga_model),
        env_(env),
        d_safe_(d_safe) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataCollisionCacheGA<Scalar>*>(data.get());
    
    // Compute and cache witness geometry for all collision pairs.
    computeDistanceCache(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    // Set residual as (d_safe - distance) for each collision pair
    // Activation model will handle max(0, r) to create barrier
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->r(i) = d_safe_ - d->env_data.distance[i];
    }
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataCollisionCacheGA<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    // Reuse the witness geometry cached in calc().
    higherKinematics(ga_model_, *(d->ga_data), x.head(nq));

    computeDistanceJacobianCache(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    data->Rx.setZero();
    data->Ru.setZero();
    
    // Set Jacobian for each collision pair
    // Negative sign because residual is (d_safe - distance)
    // dr/dq = -d(distance)/dq
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->Rx.row(i).head(nq) = -d->env_data.jac_dist[i];
    }
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataCollisionCacheGA<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }
  const Environment<Scalar>& get_environment() const { return env_; }
  Scalar get_d_safe() const { return d_safe_; }

 private:
  Model<Scalar> ga_model_;
  Environment<Scalar> env_;
  Scalar d_safe_;  // Safety distance threshold
};
