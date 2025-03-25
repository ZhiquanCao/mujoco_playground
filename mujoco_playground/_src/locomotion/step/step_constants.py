# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for STEP Humanoid."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "step"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]

NUM_DOF = 10
# two parameters for the randomize script
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 2

FEET_SITES = [
    "left_foot",
    "right_foot",
]

# HAND_SITES = [
#     "left_palm",
#     "right_palm",
# ]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "root"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-0.349066, 0.349066),  # hip yaw
    (-0.349066, 0.349066),  # hip roll
    (-1.5708, 0.349066),    # hip pitch
    (-0.698132, 0),         # knee
    (-0.698132, 0.698132),  # ankle

    # Right leg.
    (-0.349066, 0.349066),  # hip yaw
    (-0.349066, 0.349066),  # hip roll
    (-0.349066, 1.5708),    # hip pitch
    (0, 0.698132),          # knee
    (-0.698132, 0.698132),  # ankle
)