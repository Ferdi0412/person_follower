#!/usr/bin/env python3
"""Subscribes to the hand pose message, to classify gestures based on them."""
import numpy as np

import rospkg
import rospy
from std_msgs.msg import String
from person_follower_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray, Marker

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional
import math
import json

################
### SETTINGS ###
################
# Values in degrees
BUFFER = 20
OPEN_THRESHOLD = 130
FOLDED_THRESHOLD = 90

# Frames, meters
MOTION_WINDOW = 8
MOTION_THRESHOLD = 0.05

#################
### CONSTANTS ###
#################
WRIST = 0
THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP  = 1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

FINGER_IDXS = {
    "thumb":  (WRIST,      THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP),
    "index":  (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
    "pinky":  (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP)
}

FINGER_JOINTS = {
    "thumb":  (0,  3,  4), # (2,  3,  4),
    "index":  (5,  6,  8),
    "middle": (9,  10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20)
}

class FingerState(Enum):
    BUFFER = auto()
    OPEN   = auto()
    FOLDED = auto()

class Direction(Enum):
    UP      = auto()
    DOWN    = auto()
    LEFT    = auto()
    RIGHT   = auto()
    TOWARDS = auto()
    AWAY    = auto()
    STATIC  = auto()

AXES = (
    (Direction.RIGHT,   np.array([1.0,  0.0,  0.0])),
    (Direction.LEFT,    np.array([-1.0, 0.0,  0.0])),
    (Direction.DOWN,    np.array([0.0,  1.0,  0.0])),
    (Direction.UP,      np.array([0.0, -1.0,  0.0])),
    (Direction.AWAY,    np.array([0.0,  0.0,  1.0])),
    (Direction.TOWARDS, np.array([0.0,  0.0, -1.0]))
)

################
### GEOMETRY ###
################
def joint_angle_deg(a, b, c):
    v1 = a-b
    v2 = c-b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

#############################
### STATE CLASSIFICATIONS ###
#############################
def classify_direction(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return Direction.UP
    v /= norm

    best_dir, best_dot = Direction.UP, 0.708
    for d, axis in AXES:
        dv = float(np.dot(v, axis))
        if dv > best_dot:
            best_dot = dv
            best_dir = d
    return best_dir

def classify_finger(angle, prev_class=None):
    if angle >= OPEN_THRESHOLD:
        return FingerState.OPEN

    if angle <= FOLDED_THRESHOLD:
        return FingerState.FOLDED

    if prev_class == FingerState.OPEN and angle >= OPEN_THRESHOLD - BUFFER:
        return FingerState.OPEN

    if prev_class == FingerState.FOLDED and angle <= FOLDED_THRESHOLD + BUFFER:
        return FingerState.FOLDED

    return FingerState.BUFFER

def classify_motion(d):
    if np.linalg.norm(d) < MOTION_THRESHOLD:
        return Direction.STATIC
    return classify_direction(d)

#####################
### SAMPLING UTIL ###
#####################
@dataclass
class HandSample:
    is_right: bool
    palm_dir: Direction
    thumb_dir: Direction
    knuckle_dir: Direction
    finger_states: Dict[str, FingerState]

def build_hand_sample(pose_msg, prev_sample=None):
    is_right = pose_msg.score > 0.5

    pts = np.array([[k.x, k.y, k.z] for k in pose_msg.keypoints], dtype=np.float64)

    finger_states = {}
    for name, (a, b, c) in FINGER_JOINTS.items():
        prev = prev_sample.finger_states[name] if prev_sample else None
        angle = joint_angle_deg(pts[a], pts[b], pts[c])
        finger_states[name] = classify_finger(angle, prev)

    wrist = pts[WRIST]

    normal = np.cross(pts[INDEX_MCP] - wrist, pts[PINKY_MCP] - wrist)
    if not is_right:
        normal = -normal

    return HandSample(
        is_right,
        classify_direction(normal),
        classify_direction(pts[THUMB_CMC] - wrist),
        classify_direction(pts[INDEX_MCP] - wrist),
        finger_states
    )

##############
### VOTING ###
##############
def majority(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

class HandFilter:
    """Use voting to select hand state - direction and fingers."""
    def __init__(self, window_len):
        self._history = deque(maxlen=window_len)

    def update(self, sample):
        self._history.append(sample)
        recent = list(self._history)
        palm_dir = majority([s.palm_dir for s in recent])
        thumb_dir = majority([s.thumb_dir for s in recent])
        knuckle_dir = majority([s.knuckle_dir for s in recent])
        finger_states = {f: majority([s.finger_states[f] for s in recent]) for f in FINGER_JOINTS}
        return HandSample(sample.is_right, palm_dir, thumb_dir, knuckle_dir, finger_states)

class MotionFilter:
    """Use first and last positions in window to determine motion."""
    def __init__(self, window_len=MOTION_WINDOW):
        self._history = deque(maxlen=window_len)

    def update(self, wrist_pos):
        self._history.append(wrist_pos)
        if len(self._history) < 2:
            return Direction.STATIC
        return classify_motion(self._history[-1] - self._history[0])

########################
### Gesture Template ###
########################
@dataclass
class GestureTemplate:
    name: str
    hand: str = "either"
    finger_states: Dict[str, FingerState] = field(default_factory=dict)
    palm_dir: Optional[Direction] = None
    thumb_dir: Optional[Direction] = None
    knuckle_dir: Optional[Direction] = None
    motion: Optional[Direction] = None

    def matches(self, hand_sample, sampled_motion):
        for finger, constraint in self.finger_states.items():
            if isinstance(constraint, list):
                if hand_sample.finger_states[finger] not in constraint:
                    return False
            else:
                if hand_sample.finger_states[finger] != constraint:
                    return False

        if self.palm_dir and self.palm_dir != hand_sample.palm_dir:
            return False

        if self.knuckle_dir and self.knuckle_dir != hand_sample.knuckle_dir:
            return False

        if self.thumb_dir and self.thumb_dir != hand_sample.thumb_dir:
            return False

        if self.motion and sampled_motion != self.motion:
            return False

        return True

    @staticmethod
    def from_json(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return GestureTemplate.from_config(json.load(file))

    @staticmethod
    def from_config(values):
        if isinstance(values, dict):
            names = list(values.keys())
            values = [values[n] for n in names]
            for i, n in enumerate(names):
                values[i]["name"] = n

        templates = []
        for v in values:
            templates.append(GestureTemplate(
                name = v["name"],
                hand = v.get("hand", "either"),
                finger_states = GestureTemplate.parse_fingers(v["finger_states"]),
                palm_dir      = GestureTemplate.parse_direction(v.get("palm_dir")),
                thumb_dir     = GestureTemplate.parse_direction(v.get("thumb_dir")),
                knuckle_dir   = GestureTemplate.parse_direction(v.get("knuckle_dir")),
                motion        = GestureTemplate.parse_direction(v.get("motion"))
            ))

        return templates

    @staticmethod
    def parse_fingers(fingers):
        finger_states = {}
        for k, v in fingers.items():
            if v.lower() in ["open", "o"]:
                finger_states[k] = FingerState.OPEN
            elif v.lower() in ["folded", "f"]:
                finger_states[k] = FingerState.FOLDED
            else:
                raise ValueError(f"Unrecognized finger state '{v}' for finger '{k}'!")
        return finger_states

    @staticmethod
    def parse_direction(direction):
        if direction is None:
            return None

        direction = direction.lower()
        if direction in ["towards", "t"]:
            return Direction.TOWARDS
        if direction in ["away", "a"]:
            return Direction.AWAY
        if direction in ["up", "u"]:
            return Direction.UP
        if direction in ["down", "d"]:
            return Direction.DOWN
        if direction in ["left", "l"]:
            return Direction.LEFT
        if direction in ["right", "r"]:
            return Direction.RIGHT
        if direction in ["static", "s"]:
            return Direction.STATIC
        if direction is None:
            return None
        raise ValueError(f"Unrecognized direction '{direction}'!")

        

###########################
### SLIDING WINDOW VOTE ###
###########################
class VoteFilter:
    def __init__(self, window_len=5):
        self._history = deque(maxlen=window_len)

    def update(self, value):
        self._history.append(value)
        return self.vote()

    def vote(self):
        if not self._history:
            return
        counts = {}
        for v in self._history:
            counts[v] = counts.get(v, 0) + 1
        best_count = max(counts.values())
        for v in reversed(self._history):
            if counts[v] == best_count:
                return v

    def reset(self):
        self._history.clear()

################
### ROS NODE ###
################
class GestureNode:
    def __init__(self, gestures):
        rospy.init_node("person_follower_gesture_node")

        self.gestures = gestures
        self.none_gesture = ""

        self.motion_filters  = {"left": MotionFilter(), "right": MotionFilter()}
        self.hand_filters = {"left": HandFilter(5), "right": HandFilter(5)}
        self.vote_filter  = VoteFilter(5)
        self.last_published = None

        self.gesture_pub = rospy.Publisher("/person_follower/gesture", String, queue_size=1)
        self.vis_pub = rospy.Publisher("/person_follower/gesture/visualization", MarkerArray, queue_size=1)

        self.hands_sub = rospy.Subscriber("/person_follower/hands", PoseArray, self.hands_callback, queue_size=5)

    def classify(self, samples_by_side, motions_by_side):
        for g in self.gestures:
            for side in (("right", "left") if g.hand == "either" else (g.hand,)):
                sample = samples_by_side.get(side)
                if sample and g.matches(sample, motions_by_side.get(side)):
                    return g.name
        return self.none_gesture

    def hands_callback(self, msg):
        smoothed, motions, poses = {}, {}, {}
        for pose in msg.poses:
            is_right = pose.score > 0.5
            side = "right" if is_right else "left"

            prev_filtered = (
                self.hand_filters[side]._history[-1]
                if self.hand_filters[side]._history
                else None
            )

            raw = build_hand_sample(pose, prev_filtered)
            smoothed[side] = self.hand_filters[side].update(raw)
            motions[side]  = self.motion_filters[side].update(np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]))
            poses[side] = pose

        raw_gesture = self.classify(smoothed, motions)
        voted_gesture = self.vote_filter.update(raw_gesture)

        self.gesture_pub.publish(String(data=voted_gesture))
        self.last_published = voted_gesture

        self.visualize(smoothed, poses, voted_gesture, msg.header)
        # rospy.loginfo("Gesture '%s'", voted_gesture)


    def visualize(self, smoothed, poses, voted_gesture, header):
        if not self.vis_pub.get_num_connections():
            return

        marker_array = MarkerArray()
        for i, key in enumerate(smoothed):
            state = smoothed[key].finger_states
            pose = poses[key]

            for j, pt in enumerate(pose.keypoints):
                marker = Marker()
                marker.header = header
                marker.lifetime = rospy.Duration(0.5) 

                marker.id = i * (21 + 1) + j
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                marker.pose.position.x = pt.x + 0.4 * (1 if key == 'right' else -1)
                marker.pose.position.y = pt.y
                marker.pose.position.z = pt.z
                
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1

                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02

                marker.color.a = 1.0

                for f, idxs in FINGER_IDXS.items():
                    if j in idxs:
                        finger = f
                        break
                else:
                    finger = "thumb"

                if state[finger] == FingerState.FOLDED:
                    marker.color.r = 0.8
                elif state[finger] == FingerState.OPEN:
                    marker.color.g = 0.8
                else:
                    marker.color.b = 0.8

                marker_array.markers.append(marker)

            label = Marker()
            label.header = header
            label.lifetime = rospy.Duration(0.5)

            label.id = marker_array.markers[-1].id + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.text = voted_gesture or ""

            label.pose.position.x = 0.4 if key == "right" else -0.4
            label.pose.position.y = -0.4
            label.pose.position.z = 0

            label.pose.orientation.x = 0
            label.pose.orientation.y = 0
            label.pose.orientation.z = 0
            label.pose.orientation.w = 1

            label.scale.z = 0.05

            label.color.a = 1.0
            label.color.g = 0.6
            label.color.r = 0.2

            marker_array.markers.append(label)

        self.vis_pub.publish(marker_array)




###################
### ENTRY POINT ###
###################
if __name__ == "__main__":
    # GESTURES = [
    #     GestureTemplate(
    #         name="good", hand="either",
    #         finger_states={"index": FingerState.FOLDED, "middle": FingerState.FOLDED,
    #                        "ring":  FingerState.FOLDED, "pinky":  FingerState.FOLDED,
    #                        "thumb": FingerState.OPEN},
    #             thumb_dir=Direction.UP
    #     ),
    #     GestureTemplate(
    #         name="follow-me", hand="either",
    #         finger_states={"index": FingerState.OPEN, "middle": FingerState.OPEN,
    #                         "ring": FingerState.OPEN, "pinky": FingerState.OPEN},
    #         knuckle_dir=Direction.UP, palm_dir=Direction.AWAY, motion=Direction.AWAY
    #     ),
    #     GestureTemplate(
    #         name="stop", hand="either",
    #         finger_states={"index": FingerState.FOLDED, "middle": FingerState.FOLDED,
    #                         "ring": FingerState.FOLDED, "pinky": FingerState.FOLDED},
    #         knuckle_dir=Direction.UP, palm_dir=Direction.TOWARDS, motion=Direction.STATIC
    #     ),
    #     GestureTemplate(
    #         name="move-back", hand="either",
    #         finger_states={"index": FingerState.OPEN, "middle": FingerState.OPEN,
    #                         "ring": FingerState.OPEN, "pinky": FingerState.OPEN},
    #         knuckle_dir=Direction.UP, palm_dir=Direction.TOWARDS, motion=Direction.TOWARDS
    #     ),
    #     GestureTemplate(
    #         name="move-forward", hand="either",
    #         finger_states={"index": FingerState.OPEN, "middle": FingerState.OPEN,
    #                         "ring": FingerState.OPEN, "pinky": FingerState.OPEN},
    #         knuckle_dir=Direction.UP, palm_dir=Direction.TOWARDS, motion=Direction.AWAY
    #     ),
    # ]

    rospack = rospkg.RosPack()
    package_root = rospack.get_path("person_follower")

    gestures = GestureTemplate.from_json(package_root + "/scripts/gestures.json")

    print(gestures)

    node = GestureNode(gestures)
    rospy.spin()