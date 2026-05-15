#include <cmath>
#include <functional>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
class PedestrianPatrolPlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    this->model = _model;
    this->world = _model->GetWorld();
    this->startPose = _model->WorldPose();
    this->endPose = this->startPose;
    this->speed = 0.4;
    this->pauseDuration = 1.0;
    this->forward = true;
    this->isPaused = false;

    if (_sdf->HasElement("start"))
    {
      this->startPose = _sdf->Get<ignition::math::Pose3d>("start");
    }

    if (_sdf->HasElement("end"))
    {
      this->endPose = _sdf->Get<ignition::math::Pose3d>("end");
    }
    else
    {
      this->endPose.Pos().X() = this->startPose.Pos().X() + 4.0;
    }

    if (_sdf->HasElement("speed"))
    {
      this->speed = _sdf->Get<double>("speed");
    }

    if (_sdf->HasElement("pause"))
    {
      this->pauseDuration = _sdf->Get<double>("pause");
    }

    this->startPose.Pos().Z() = this->model->WorldPose().Pos().Z();
    this->endPose.Pos().Z() = this->startPose.Pos().Z();
    this->travelAxis = this->endPose.Pos() - this->startPose.Pos();
    this->travelAxis.Z() = 0.0;

    if (this->travelAxis.Length() < 1e-6)
    {
      this->travelAxis = ignition::math::Vector3d(1.0, 0.0, 0.0);
    }

    this->travelDirection = this->travelAxis;
    this->travelDirection.Normalize();
    this->travelLength = this->travelAxis.Length();
    this->currentDistance = 0.0;
    this->yawForward = std::atan2(this->travelDirection.Y(), this->travelDirection.X());
    this->yawBackward = std::atan2(-this->travelDirection.Y(), -this->travelDirection.X());

    ignition::math::Pose3d initialPose = this->startPose;
    initialPose.Rot() = ignition::math::Quaterniond(0.0, 0.0, this->yawForward);
    this->model->SetWorldPose(initialPose);

    this->lastSimTime = this->world->SimTime();
    this->pauseStartTime = this->lastSimTime;
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&PedestrianPatrolPlugin::OnUpdate, this));
  }

private:
  void OnUpdate()
  {
    const common::Time currentTime = this->world->SimTime();
    const double dt = (currentTime - this->lastSimTime).Double();
    if (dt <= 0.0)
    {
      return;
    }
    this->lastSimTime = currentTime;

    if (this->isPaused)
    {
      if ((currentTime - this->pauseStartTime).Double() >= this->pauseDuration)
      {
        this->isPaused = false;
        this->forward = !this->forward;
      }
      else
      {
        return;
      }
    }

    const double step = this->speed * dt;

    if (this->forward)
    {
      this->currentDistance += step;
      if (this->currentDistance >= this->travelLength)
      {
        this->currentDistance = this->travelLength;
        this->ApplyPose(this->yawForward);
        this->isPaused = true;
        this->pauseStartTime = currentTime;
        return;
      }
      this->ApplyPose(this->yawForward);
      return;
    }

    this->currentDistance -= step;
    if (this->currentDistance <= 0.0)
    {
      this->currentDistance = 0.0;
      this->ApplyPose(this->yawBackward);
      this->isPaused = true;
      this->pauseStartTime = currentTime;
      return;
    }

    this->ApplyPose(this->yawBackward);
  }

  void ApplyPose(double yaw)
  {
    ignition::math::Pose3d pose = this->startPose;
    pose.Pos() += this->travelDirection * this->currentDistance;
    pose.Pos().Z() = this->startPose.Pos().Z();
    pose.Rot() = ignition::math::Quaterniond(0.0, 0.0, yaw);
    this->model->SetWorldPose(pose);
  }

  physics::ModelPtr model;
  physics::WorldPtr world;
  event::ConnectionPtr updateConnection;
  ignition::math::Pose3d startPose;
  ignition::math::Pose3d endPose;
  ignition::math::Vector3d travelAxis;
  ignition::math::Vector3d travelDirection;
  common::Time lastSimTime;
  common::Time pauseStartTime;
  double speed;
  double pauseDuration;
  double travelLength;
  double currentDistance;
  double yawForward;
  double yawBackward;
  bool forward;
  bool isPaused;
};

GZ_REGISTER_MODEL_PLUGIN(PedestrianPatrolPlugin)
}  // namespace gazebo
