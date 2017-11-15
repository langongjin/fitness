/*
* Copyright (C) 2017 Vrije Universiteit Amsterdam
*
* Licensed under the Apache License, Version 2.0 (the "License");
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Description: TODO: <Add brief description about file purpose>
* Author: Elte Hupkes
* Date: May 9, 2015
*
*/

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <boost/make_shared.hpp>

#include <revolve/gazebo/motors/Motor.h>
#include <revolve/gazebo/sensors/VirtualSensor.h>

#include "brain/GenericLearnerBrain.h"
#include "brain/Helper.h"
#include "brain/MLMPCPGBrain.h"
#include "brain/HyperNEAT_CPPN.h" //HyperNEAT_CPG
#include "brain/HyperNEAT_Splines.h"
#include "brain/HyperNEAT_MlmpCPG.h"
#include "brain/RLPower_Splines.h"
#include "brain/RLPower_CPPN.h"
#include "brain/SUPGBrain.h"
#include "brain/SUPGBrainPhototaxis.h"
#include "brain/YamlBodyParser.h"
#include "brain/supg/SUPGGenomeManager.h"
#include "brain/learner/HyperAccNEATLearner_CPGController.h"

#include "RobotController.h"

#define pi 3.14159265

using namespace tol;

const char *getVARenv(const char *var_name)
{
  const char *env_p = std::getenv(var_name);
  if (env_p)
  {
    std::cout << "ENV " << var_name << " is: " << env_p << std::endl;
  }
  else
  {
    std::cout << "ENV " << var_name << " not found, using default value: ";
  }
  return env_p;
}

NEAT::GeneticSearchType getGeneticSearchType(const std::string &value)
{
  if ("PHASED" == value)
  {
    return NEAT::GeneticSearchType::PHASED;
  }

  if ("BLENDED" == value)
  {
    return NEAT::GeneticSearchType::BLENDED;
  }

  if ("COMPLEXIFY" == value)
  {
    return NEAT::GeneticSearchType::COMPLEXIFY;
  }

  // default value
  return NEAT::GeneticSearchType::PHASED;
}

const char *getGeneticSearchType(const NEAT::GeneticSearchType value)
{
  switch (value)
  {
    case NEAT::GeneticSearchType::BLENDED:
      return "NEAT::GeneticSearchType::BLENDED";
    case NEAT::GeneticSearchType::PHASED:
      return "NEAT::GeneticSearchType::PHASED";
    case NEAT::GeneticSearchType::COMPLEXIFY:
      return "NEAT::GeneticSearchType::COMPLEXIFY";
    default:
      return "undefined";
  }
}

void init_asyncneat(
        const std::string &robot_name,
        std::unique_ptr< NEAT::GenomeManager > custom_genome_manager)
{
  if (custom_genome_manager)
  {
    AsyncNeat::Init(std::move(custom_genome_manager));
  }
  else
  {
    AsyncNeat::Init(robot_name);
  }
  size_t populationSize = 10;
  NEAT::real_t mutate_add_node_prob = 0.01;
  NEAT::real_t mutate_add_link_prob = 0.3;
  NEAT::GeneticSearchType geneticSearchType =
          NEAT::GeneticSearchType::COMPLEXIFY;

  if (const char *env_p = getVARenv("NEAT_POP_SIZE"))
  {
    try
    {
      populationSize = (size_t)std::stoul(env_p);
    } catch (const std::invalid_argument &e)
    {
      std::cout
              << "ERROR DECODING STRING \"NEAT_POP_SIZE\" to unsigned long:"
              << " using default value " << populationSize
              << " instead" << std::endl;
    }
  }
  else
  {
    std::cout << populationSize << std::endl;
  }

  if (const char *env_p = getVARenv("NEAT_MUTATE_ADD_NODE_PROB"))
  {
    try
    {
      mutate_add_node_prob = (float)std::stod(env_p);
    } catch (const std::invalid_argument &e)
    {
      std::cout
              << "ERROR DECODING STRING \"NEAT_MUTATE_ADD_NODE_PROB\" "
              << "to double: using default value " << mutate_add_node_prob
              << " instead" << std::endl;
    }
  }
  else
  {
    std::cout << mutate_add_node_prob << std::endl;
  }

  if (const char *env_p = getVARenv("NEAT_MUTATE_ADD_LINK_PROB"))
  {
    try
    {
      mutate_add_link_prob = (float)std::stod(env_p);
    } catch (const std::invalid_argument &e)
    {
      std::cout << "ERROR DECODING STRING \"NEAT_MUTATE_ADD_LINK_PROB\" "
                << "to double: using default value " << mutate_add_link_prob
                << " instead" << std::endl;
    }
  }
  else
  {
    std::cout << mutate_add_link_prob << std::endl;
  }

  if (const char *env_p = getVARenv("NEAT_SEARCH_TYPE"))
  {
    geneticSearchType = getGeneticSearchType(env_p);
  }
  else
  {
    std::cout << getGeneticSearchType(geneticSearchType) << std::endl;
  }
  // 10 - 25 - 50 - 75 - 100 - 1000
  AsyncNeat::SetPopulationSize(populationSize);
  AsyncNeat::SetMutateAddNodeProb(mutate_add_node_prob);
  AsyncNeat::SetMutateAddLinkProb(mutate_add_link_prob);
  std::cout << "Setting up genetic search type to: "
            << getGeneticSearchType(geneticSearchType) << std::endl;
  AsyncNeat::SetSearchType(geneticSearchType);
}

RobotController::RobotController()
{
}

RobotController::~RobotController()
{
  AsyncNeat::CleanUp();
}

void RobotController::Load(
        ::gazebo::physics::ModelPtr _parent, //// ::表示顶层命名空间(全局变量）
        sdf::ElementPtr _sdf)
{ //// ::表示顶层命名空间(全局变量）std::string -> ::std::string 这样也可以
  //// ::和 文件路径的 / 可以对照理解, linux下面没有盘符之分,只有一个盘,/usr/share/abc.txt 第一个/就代表硬盘根目录
  ::revolve::gazebo::RobotController::Load(_parent, _sdf);
  std::cout << "ToL Robot loaded." << std::endl;
}

void RobotController::LoadBrain(sdf::ElementPtr sdf)
{
  try
  {
    evaluator_ = boost::make_shared< Evaluator >();
    const std::string &robot_name = this->model->GetName();

    size_t motor_n = 0;  // motors_.size();
    for (const auto &motor : motors_)
    {
      motor_n += motor->outputs();
    }
    size_t sensor_n = 0;  // sensors_.size();
    for (const auto &sensor : sensors_)
    {
      sensor_n += sensor->inputs();
    }

    if (not sdf->HasElement("rv:brain"))
    {
      std::cerr << "No robot brain detected, this is probably an error."
                << std::endl;
      return;
    }
    auto brain = sdf->GetElement("rv:brain");

    if (not brain->HasAttribute("algorithm"))
    {
      std::cerr << "Brain does not define type, this is probably an error."
                << std::endl;
      return;
    }

    std::string brainType = brain->GetAttribute("algorithm")->GetAsString();
    if ("rlpower::spline" == brainType)
    {
      brain_.reset(new tol::RLPower_Splines(
              robot_name,
              brain,
              evaluator_,
              motors_));
    }
    else if ("rlpower::net" == brainType)
    {
      brain_.reset(new tol::RLPower_CPG(
              robot_name,
              brain,
              evaluator_,
              motors_,
              sensors_));
    }
    else if ("hyperneat::net" == brainType)
    {
      brain_.reset(new tol::HyperNEAT_CPG(
              robot_name,
              brain,
              evaluator_,
              motors_,
              sensors_));
    }
    else if ("rafhyperneat::mlmp_cpg" == brainType)
    {
      brain_.reset(new tol::HyperNEAT_MlmpCPG(
              robot_name,
              brain,
              evaluator_,
              motors_,
              sensors_));
    }
    else if ("hyperneat::spline" == brainType)
    {
      brain_.reset(new tol::HyperNEAT_Splines(
              robot_name,
              brain,
              evaluator_,
              motors_,
              sensors_));
    }
    else if ("rlpower::mlmp_cpg" == brainType)
    {
      brain_.reset(new tol::MlmpCPGBrain(
              robot_name,
              evaluator_,
              motor_n,
              sensor_n));
    }
    else if ("hyperneat::mlmp_cpg" == brainType)
    {
      init_asyncneat(robot_name, std::unique_ptr< NEAT::GenomeManager >());
      auto modelName = this->model->GetName();
      tol::YamlBodyParser *parser = new tol::YamlBodyParser();
      modelName = modelName.substr(0, modelName.find("-")) + ".yaml";
      parser->parseFile(modelName);
      auto connections = parser->connections();
      auto cpgs_coordinates = parser->coordinates();
      brain_.reset(new tol::GenericLearnerBrain(
              new revolve::brain::HyperAccNEATLearner_CPGController(
                      robot_name,
                      evaluator_,
                      sensor_n,
                      motor_n,
                      2,  // coordinates cardinality
                      connections,
                      cpgs_coordinates,
                      30,  // seconds
                      999)));  //  -1 // infinite evaluations
      delete parser;
    }
    else if ("supg::phototaxis" == brainType)
    {
      init_asyncneat(robot_name, std::unique_ptr< NEAT::GenomeManager >(
              new SUPGGenomeManager(robot_name)));

      std::vector< std::vector< float > > coordinates;

      const std::string robot_type_str = getVARenv("ROBOT_TYPE");
      const Helper::RobotType robot_type =
              Helper::parseRobotType(robot_type_str);
      std::cout << "Loading SUPG configuration for robot " << robot_type
                << std::endl;
      switch (robot_type)
      {
        case Helper::spider9:
          // SPIDER 9
          //     #
          //     #
          // # # O # #
          //     #
          //     #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg01Joint
                           {1,    0},  //    1},
                           {.5,   0},  //    -1},
                           // Leg10Joint Leg11Joint
                           {-1,   0},  //    1},
                           {-.5f, 0},  //    -1},
                           // Leg20Joint Leg21Joint
                           {0,    1},  //    1},
                           {0,    .5},  //   -1},
                           // Leg30Joint Leg31Joint
                           {0,    -1},  //   1},
                           {0,    -.5f}  // , -1}
                   });
          break;
        case Helper::spider13:
          // SPIDER 13
          //       #
          //       #
          //       #
          // # # # O # # #
          //       #
          //       #
          //       #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg01Joint Leg02Joint
                           {.333,   0},
                           {.666,   0},
                           {1,      0},
                           // Leg10Joint Leg11Joint Leg12Joint
                           {-.333f, 0},
                           {-.666f, 0},
                           {-1,     0},
                           // Leg20Joint Leg21Joint Leg22Joint
                           {0,      .333},
                           {0,      .666},
                           {0,      1},
                           // Leg30Joint Leg31Joint Leg32Joint
                           {0,      -.333f},
                           {0,      -.666f},
                           {0,      -1},
                   });
          break;
        case Helper::spider17:
          // SPIDER 17
          //         #
          //         #
          //         #
          //         #
          // # # # # O # # # #
          //         #
          //         #
          //         #
          //         #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg01Joint Leg02Joint
                           {.25,   0},
                           {.5,    0},
                           {.75,   0},
                           {1,     0},
                           // Leg10Joint Leg11Joint Leg12Joint
                           {-.25f, 0},
                           {-.5f,  0},
                           {-.75f, 0},
                           {-1,    0},
                           // Leg20Joint Leg21Joint Leg22Joint
                           {0,     .25},
                           {0,     .5},
                           {0,     .75},
                           {0,     1},
                           // Leg30Joint Leg31Joint Leg32Joint
                           {0,     -.25f},
                           {0,     -.5f},
                           {0,     -.75f},
                           {0,     -1},
                   });
          break;
        case Helper::gecko7:
          // GECKO 5
          // #   #
          // O # #
          // #   #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint
                           {-1,   +1},
                           // Leg01Joint
                           {-1,   -1},
                           // BodyJoint0
                           {-.5f, 0},
                           // BodyJoint1
                           {+.5f, 0},
                           // Leg10Joint
                           {+1,   +1},
                           // Leg11Joint
                           {+1,   -1},
                   });
          break;
        case Helper::gecko12:
          // GECKO 12
          // #     #
          // #     #
          // O # # #
          // #     #
          // #     #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg001Joint
                           {-1.0f, +0.5f},
                           {-1,    +1},
                           // Leg01Joint Leg011Joint
                           {-1.0f, -0.5f},
                           {-1,    -1},
                           // BodyJoint0 BodyJoint1 BodyJoint2
                           {-.5f,  0},
                           {0,     0},
                           {+.5f,  0},
                           // Leg10Joint Leg101Joint
                           {+1,    +0.5f},
                           {+1,    +1},
                           // Leg11Joint Leg111Joint
                           {+1,    -0.5f},
                           {+1,    -1},
                   });
          break;
        case Helper::gecko17:
          // GECKO 17
          // #     #
          // #     #
          // O # # #
          // #     #
          // #     #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg001Joint Leg002Joint
                           {-1.0f,  +.333f},
                           {-1.0f,  +.666f},
                           {-1,     +1},
                           // Leg01Joint Leg011Joint Leg012Joint
                           {-1.0f,  -.333f},
                           {-1.0f,  -.333f},
                           {-1,     -1},
                           // BodyJoint0 BodyJoint1 BodyJoint2 BodyJoint3
                           {-.666f, 0},
                           {-.333f, 0},
                           {+.333f, 0},
                           {+.666f, 0},
                           // Leg10Joint Leg101Joint Leg102Joint
                           {+1,     +.333f},
                           {+1,     +.666f},
                           {+1,     +1},
                           // Leg11Joint Leg111Joint Leg112Joint
                           {+1,     -.333f},
                           {+1,     -.666f},
                           {+1,     -1},
                   });
          break;
        case Helper::snake5:
          // SNAKE 5
          //
          // # # O # #
          //
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint
                           {-.5f, 0},
                           // Leg01Joint
                           {-1,   0},
                           // Leg10Joint
                           {+.5f, 0},
                           // Leg11Joint
                           {+1,   0},
                   });
          break;
        case Helper::snake7:
          // SNAKE 7
          //
          // # # # O # # #
          //
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint
                           {-.333f, 0},
                           // Leg01Joint
                           {-.666f, 0},
                           // Leg02Joint
                           {-1,     0},
                           // Leg10Joint
                           {+.333f, 0},
                           // Leg11Joint
                           {+.666f, 0},
                           // Leg12Joint
                           {+1,     0},
                   });
          break;
        case Helper::snake9:
          // SNAKE 9
          //
          // # # # # O # # # #
          //
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint
                           {-.25f, 0},
                           // Leg01Joint
                           {-.50f, 0},
                           // Leg02Joint
                           {-.75f, 0},
                           // Leg03Joint
                           {-1,    0},
                           // Leg10Joint
                           {+.25f, 0},
                           // Leg11Joint
                           {+.50f, 0},
                           // Leg12Joint
                           {+.75f, 0},
                           // Leg13Joint
                           {+1,    0},
                   });
          break;
        case Helper::babyA:
          // BABY 1
          // #
          // #   #
          // O # #
          // #   #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint
                           {-1.0f, +1},
                           // Leg01Joint
                           {-1.0f, -.3f},
                           // Leg011Joint
                           {-1.0f, -.6f},
                           // Leg021Joint
                           {-1.0f, -1.0f},
                           // BodyJoint0
                           {-.5f,  0},
                           // BodyJoint1
                           {+.5f,  0},
                           // Leg10Joint
                           {+1,    +1},
                           // Leg11Joint
                           {+1,    -1},
                   });
          break;
        case Helper::babyB:
          // BABY 2
          //
          //       #
          // # # # O # # #
          //       #
          //       #
          //       #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg01Joint Leg02Joint
                           {1,     0},
                           {.666f, 0},
                           {.333f, 0},
                           // Leg10Joint
                           {-1,    0},
                           // Leg20Joint Leg21Joint Leg22Joint
                           {0,     1},
                           {0,     .666f},
                           {0,     .333f},
                           // Leg30Joint Leg31Joint Leg32Joint
                           {0,     -1},
                           {0,     -.666f},
                           {0,     .333f},
                   });
          break;
        case Helper::babyC:
          // BABY 3
          // #       #
          // #       x
          // #       #
          // O # # # #
          // #       #
          // #       #
          // #       #
          coordinates = std::vector< std::vector< float > >
                  ({
                           // Leg00Joint Leg001Joint Leg002Joint
                           {-1.0f,  +.333f},
                           {-1.0f,  +.666f},
                           {-1.0f,  +1},
                           // Leg01Joint Leg011Joint Leg012Joint
                           {-1.0f,  -.333f},
                           {-1.0f,  -.333f},
                           {-1.0f,  -1},
                           // BodyJoint0 BodyJoint1 BodyJoint2 BodyJoint3
                           {-.666f, 0},
                           {-.333f, 0},
                           {+.333f, 0},
                           {+.666f, 0},
                           // Leg10Joint Leg101Joint Leg102Joint
                           {+1.0f,  +.333f},
                           {+1.0f,  +.666f},
                           {+1.0f,  +1},
                           // Leg11Joint Leg111Joint Leg112Joint
                           {+1.0f,  -.333f},
                           {+1.0f,  -.666f},
                           {+1.0f,  -.9f},
                   });
          break;
      }

      // brain_.reset(
      //        new SUPGBrain(evaluator_, coordinates, motors_, sensors_));
      brain_.reset(new SUPGBrainPhototaxis(
              robot_name,
              evaluator_,
              50,
              coordinates,
              motors_,
              sensors_));
    }
    else
    {
      std::cout << "Calling default ANN brain." << std::endl;
      revolve::gazebo::RobotController::LoadBrain(sdf);
    }
  } catch (std::exception &e)
  {
    // needed because otherwise the exception dies silently and debugging is
    // a nightmare
    std::cerr << "Exception occurred while running RobotController::LoadBrain\n"
              << "exception: " << e.what() << std::endl;
  }
}

void RobotController::DoUpdate(const gazebo::common::UpdateInfo info)
{
  revolve::gazebo::RobotController::DoUpdate(info);
//  evaluator_->updatePosition(this->model->GetRelativePose().Ign());
    //TODO: NEW LINE FOR DIFFERENT LINK
  //auto bodyPose = this->model->GetRelativePose().Ign();
  //GetWorldPose(): Get the absolute pose of the entity.
  auto currentCore = this->model->GetLink("link_component_Core__box")
                      ->GetWorldPose().Ign();
  auto currentOrientation = this->model->GetLink
                  ("link_component_Core__box")->GetWorldPose().Ign();
  evaluator_->updatePosition(currentCore, currentOrientation);
//   reinterpret_cast<SUPGBrainPhototaxis&>(*brain_).updateRobotPosition(pose);
}

// EVALUATOR CODE,the constructor function have same name with the class
// name, without return, no need to call, execute automatically.
RobotController::Evaluator::Evaluator()
{
  // set the position to zero
  currentCore_.Reset();
  previousCore_.Reset();
    fitnessBest = 0;
}

void RobotController::Evaluator::start()
{
  //previousPosition_ = currentPosition_;
  previousCore_ = currentCore_;
  previousOrientation_ = currentOrientation_;
}

double RobotController::Evaluator::fitness()
{
  double lineDirection_K;
  double lineDirection_b;
  double alpha;
  double alpha0;
  double alpha1;

  std::cout << "previousCore_.Pos().X() = " << previousCore_.Pos().X()
            << ", previousCore_.Pos().Y() = "<< previousCore_.Pos().Y()
            << ", currentCore_.Pos().X() = " << currentCore_.Pos().X()
            << ", currentCore_.Pos().Y() = " << currentCore_.Pos().Y()
            << std::endl;

  alpha0 = previousOrientation_.Rot().Yaw();
  lineDirection_K = tan(alpha0);
  lineDirection_b = previousCore_.Pos().Y()
                      - lineDirection_K * previousCore_.Pos().X();
//  }

  //calculate the projected coordination of currentCore in lineDirection
  double projectionX = (lineDirection_K * (currentCore_.Pos().Y() -
          lineDirection_b) + currentCore_.Pos().X()) / (lineDirection_K *
          lineDirection_K + 1);
  double projectionY = lineDirection_K * projectionX + lineDirection_b;

  //calculate the angle (alpha) between lineDirection and currentcore

  alpha1 = atan2(currentCore_.Pos().Y() - previousCore_.Pos().Y(),
                 currentCore_.Pos().X() - previousCore_.Pos().X());
  if (std::abs(alpha1 - alpha0) > pi)
    alpha = 2 * pi - std::abs(alpha1) - std::abs(alpha0);
  else
    alpha = std::abs(alpha1 - alpha0);

  //calculate the fitnessDirection based on distProjection, alpha, penalty
  double distProjection;
  double distPenalty;
  double penalty;
  double fitnessDirection;
  double k = 5.0;
  double ksi = 1.0;
  if(alpha > 0.5 * pi)
  {
    distProjection = - std::sqrt(
            std::pow((previousCore_.Pos().X() - projectionX), 2.0) +
            std::pow((previousCore_.Pos().Y() - projectionY), 2.0));
    distPenalty = - std::sqrt(
            std::pow((currentCore_.Pos().X() - projectionX), 2.0) +
            std::pow((currentCore_.Pos().Y() - projectionY), 2.0));
    penalty = - 0.01 * distPenalty;
  }
  else
  {
    distProjection = std::sqrt(
            std::pow((previousCore_.Pos().X() - projectionX), 2.0) +
            std::pow((previousCore_.Pos().Y() - projectionY), 2.0));
    distPenalty = std::sqrt(
            std::pow((currentCore_.Pos().X() - projectionX), 2.0) +
            std::pow((currentCore_.Pos().Y() - projectionY), 2.0));
    penalty = 0.01 * distPenalty;
  }
  std::cout << "distProjection = " << distProjection
            << ", aplha = " << alpha << ", penalty = "<<penalty<<std::endl;

  fitnessDirection = 0.005 + distProjection / (k * alpha + ksi) -penalty;
  std::cout << "fitnessDirection : " << fitnessDirection << std::endl;

  //previousPosition_ = currentPosition_;
  previousCore_ = currentCore_;
  previousOrientation_ = currentOrientation_;

    if (fitnessBest < fitnessDirection)
        fitnessBest = fitnessDirection;
    std::cout << "fitnessBest = " << fitnessBest << std::endl;

  return fitnessDirection;  // dS / RLPower::FREQUENCY_RATE
}

//void RobotController::Evaluator::updatePosition(const ignition::math::Pose3d pose)
void RobotController::Evaluator::updatePosition(
        const ignition::math::Pose3d _currentCore,
        const ignition::math::Pose3d _currentOrientation)
{
    currentCore_ = _currentCore;
    previousOrientation_ = _currentOrientation;
}
