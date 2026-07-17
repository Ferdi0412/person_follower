#!/usr/bin/env python3
"""A simple state machine, defining the state transitions supported by the state_machine node."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import json

import rospkg
import rospy
from std_msgs.msg import String, UInt32
from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import CameraInfo

from person_follower_msgs.srv import EmbedRoi
from person_follower_msgs.msg import PoseArray

##############
### CONFIG ###
##############
IDLE         = "IDLE"
FOLLOW_STATE = "follow"
STOP = "stop"

FOLLOW_STANDOFF = 1.0
FOLLOW_STEP     = 0.25
VIS_THRESHOLD   = 0.65
KEEP_ALIVE      = 15

#####################
### STATE MACHINE ###
#####################
@dataclass
class StateDefinition:
    """
    name:     Name of state
    enter:    Gesture required to go from idle to this
    exit:     If used, this state latches until the corresponding gesture
    debounce: Number of repeats of `enter` to trigger state change
    release:  If used, this is number of repeats of `exit` to exit
    """
    name:     str
    enter:    str
    exit:     Optional[str] = None
    debounce: int           = 3
    release:  Optional[int] = None

    @property
    def is_momentary(self) -> bool:
        return self.exit is None

    @property
    def effective_release(self) -> int:
        return self.release or self.debounce



def _build_transitions(definitions: List[StateDefinition], idle) -> Dict[Tuple[str, str], str]:
    """Sets up all transitions to be supported. Any not contained will go to `idle`."""
    table = {}
    for d in definitions:
        table[(idle, d.enter)] = d.name
        if d.exit:
            table[(d.name, d.exit)] = idle
    return table



class StateMachine:
    """Table-driven state machine, keys are (state, gesture), values are next state"""
    def __init__(self, definitions: List[StateDefinition], idle = "IDLE"):
        self._idle_state    = idle
        self._current_state = idle

        self._table         = _build_transitions(definitions, idle)
        self._more          = {d.name: d for d in definitions}
        
        self._debounce_required = dict()
        for d in definitions:
            self._debounce_required[d.enter] = d.debounce
            if d.exit:
                self._debounce_required[d.exit] = d.effective_release

        self._last_gesture = None
        self._repeat_count = 0

    @property
    def current_state(self):
        return self._current_state

    @property
    def idle_state(self):
        return self._idle_state

    @property
    def debounced(self):
        return self._repeat_count > 3

    def has(self, gesture):
        return (self._current_state, gesture) in self._table

    def reset(self):
        self._current_state = self._idle_state
        self._last_gesture  = None
        self._repeat_count  = 0

    def update(self, gesture):
        if gesture == self._last_gesture:
            self._repeat_count += 1
        else:
            self._last_gesture = gesture
            self._repeat_count = 1

        # For well-defined state transitions
        required = self._debounce_required.get(gesture, 1)
        if self._repeat_count >= required:
            next_state = self._table.get((self._current_state, gesture))
            if next_state is not None:
                self._current_state = next_state
                return self._current_state

        # For exit-less states
        active = self._more.get(self._current_state)
        if (active is not None and active.is_momentary
                  and gesture != active.enter
                  and self._repeat_count >= active.effective_release):
            self._current_state = self._idle_state
        
        return self._current_state 

    @staticmethod
    def from_json(filename, idle = 'IDLE', debounce = 3):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            definitions = [StateDefinition(
                name     = key,
                enter    = value['enter'],
                exit     = value.get('exit')
            ) for key, value in data.items()]

        elif isinstance(data, list):
            definitions = [StateDefinition(
                name     = value['name'],
                enter    = value['enter'],
                exit     = value.get('exit')
            ) for value in data]

        else:
            raise RuntimeError(f"Malformed state json '{filename}'")

        return StateMachine(definitions, idle)

##################
### ReID Utils ###
##################
HEAD_IDXS     = (1, 2, 3, 4)
SHOULDER_IDXS = (5, 6)
HIP_IDXS      = (11, 12)
TARGET_AR     = 2.0

def reid_bbox(pose, w, h, conf_threshold=0.65):
    """Returns bounding box for ReID purposes. None if no good."""
    max_width  = min(h / TARGET_AR, w)
    max_height = min(w * TARGET_AR, h)

    kpts = pose.keypoints

    head_pts     = [kpts[i] for i in HEAD_IDXS if kpts[i].conf >= conf_threshold]
    shoulder_pts = [kpts[i] for i in SHOULDER_IDXS if kpts[i].conf >= conf_threshold]
    hip_pts      = [kpts[i] for i in HIP_IDXS if kpts[i].conf >= conf_threshold]

    if not head_pts or not shoulder_pts or not hip_pts:
        return

    torso_pts = shoulder_pts + hip_pts
    all_pts = head_pts + torso_pts

    # Get upper limit offset
    yhead     = min(pt.y for pt in head_pts)
    yshoulder = max(pt.y for pt in shoulder_pts)
    dhead     = abs(yhead - yshoulder)

    # Get lower limit offset
    yshoulder = sum(pt.y for pt in shoulder_pts) / len(shoulder_pts)
    yhip      = sum(pt.y for pt in hip_pts) / len(hip_pts)
    dhip      = abs(yhip - yshoulder)

    # Upper and lower limits
    ytop = min(pt.y for pt in all_pts) - dhead
    ybot = max(pt.y for pt in all_pts) + dhip
    cy     = (ytop + ybot) / 2.0
    height = ybot - ytop

    # Left/right
    xmin   = min(pt.x for pt in all_pts)
    xmax   = max(pt.x for pt in all_pts)
    cx     = (xmin + xmax) / 2.0
    width  = (xmax - xmin) * 2.0

    # 2:1 ratio
    if width <= 0 or height <= 0:
        return

    curr = height / width
    if curr < TARGET_AR:
        height = width * TARGET_AR
    else:
        width = height / TARGET_AR

    if width > max_width:
        width  = max_width
        height = width * TARGET_AR
    
    if height > max_height:
        height = max_height
        width  = height / TARGET_AR

    # Clamp to image limits
    cx_min = width / 2.0
    cx_max = w - width / 2.0

    cy_min = height / 2.0
    cy_max = h - height / 2.0

    cx = min(cx_max, max(cx_min, cx))
    cy = min(cy_max, max(cy_min, cy))

    return int(cx - width / 2.0), int(cy - height / 2.0), int(width), int(height)



def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return -1.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return -1.0
    
    return dot / (norm_a * norm_b)



################
### ROS NODE ###
################
class FollowStateNode:
    def __init__(self, state_machine):
        rospy.init_node("person_follower_state_node2")
        rospy.on_shutdown(self.release_target)

        rospy.loginfo("person_follower_state_node3: waiting for reid service")
        rospy.wait_for_service("get_embedding")
        rospy.loginfo("person_follower_state_node3: reid service found")

        self.state_machine = state_machine

        info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=5.0)
        self.width  = info.width
        self.height = info.height

        self.owner = None
        self.last_seen = None
        self.last_published = None

        self.pose_frame_id = None
        self.last_x = 0
        self.last_z = 0

        self.short_timeout   = rospy.Duration(0.5)
        self.target_timeout  = rospy.Duration(5.0)
        self.sim_threshold   = 0.65
        self.missing_since   = None
        self.embedding       = None
        self.known           = []

        self.state_pub  = rospy.Publisher("/person_follower/state",  String, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher("/person_follower/target", UInt32, queue_size=1, latch=True)

        self.gesture_sub = rospy.Subscriber("/person_follower/gesture", String,    self.gesture_callback, queue_size=5)
        self.pose_sub    = rospy.Subscriber("/person_follower/pose",    PoseArray, self.pose_callback,    queue_size=1)

        self.embed_client = rospy.ServiceProxy("get_embedding", EmbedRoi)
        self.goal_pub     = rospy.Publisher("/person_follower/goal", PoseStamped, queue_size=1)



    def release_target(self):
        self.owner = None
        self.embedding = None

        self.target_pub.publish(UInt32(data=0))
        self.state_pub.publish("")
        self.stop()



    def mark_missing(self):
        now = rospy.Time.now()

        if self.missing_since is None:
            self.missing_since = now
            return

        timeout = self.target_timeout if self.embedding else self.short_timeout
        if (now - self.missing_since) > timeout:
            self.release_target()



    def get_embedding(self, pose):
        """Returns (embedding, is_facing)"""   
        if not self.width or not self.height:
            return

        bbox = reid_bbox(pose, self.width, self.height)

        if bbox is None:
            return

        try:
            x, y, w, h = bbox
            resp = self.embed_client(x=x, y=y, width=w, height=h)

        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, "person_follower_state_node3: embedding failed - %s", e)
            return

        if resp.error_message:
            rospy.logwarn_throttle(5.0, "person_follower_state_node3: embedding service rejected - %s", resp.error_message)
            return

        return resp.embedding



    def mark_found(self, pose):
        self.missing_since = None
        self.last_seen     = rospy.Time.now()

        if not self.embedding:
            self.embedding = self.get_embedding(pose)
            if self.embedding:
                rospy.loginfo("person_follower_state_node2: now have embedding for %u", self.owner)



    def try_capture(self, poses):
        if not self.embedding:
            return False
        
        for p in poses:
            if p.id == 0:
                continue

            if p.id in self.known:
                continue
            
            embedding = self.get_embedding(p)
            
            if embedding:
                sim = cosine_similarity(embedding, self.embedding)
                rospy.loginfo("follow_state_node2: sim for %u and %u %.2f", p.id, self.owner, sim)

                self.known.append(p.id)
                if sim > self.sim_threshold:
                    rospy.loginfo("follow_state_nod2: new target %u", p.id)
                    self.mark_target(p.id)


    def mark_target(self, pid):
        self.target_pub.publish(UInt32(data=pid))
        self.owner     = pid
        self.last_seen = 0
        # self.embedding = None
        self.known     = []



    def gesture_callback(self, msg):
        if rospy.is_shutdown():
            return

        state = self.state_machine.update(msg.data)

        # If state is idle, but the gesture is not none - forward it
        if state == self.state_machine.idle_state and len(msg.data) and self.state_machine.debounced:
            state = msg.data

        # Only publish on change - latching
        if state != self.last_published:
            self.state_pub.publish(String(data=state))
            self.last_published = state
            self.last_x = 0
            self.last_z = 0
            rospy.loginfo("person_follower_state_node: state is now '%s'", state)



    def pose_callback(self, msg):
        # self.target_pub.publish(UInt32(data=self.owner))
        self.pose_frame_id = msg.header.frame_id

        if self.owner in [None, 0]:
            self.select_facing(msg)
            return

        for p in msg.poses:
            if p.id != self.owner:
                continue

            self.mark_found(p)

            if self.state_machine.current_state == FOLLOW_STATE:
                self.follow(p.pose)
                return

            if self.state_machine.current_state == STOP:
                self.stop()
                return

            if self.state_machine.current_state == IDLE:
                if not self.is_facing(p):
                    self.mark_missing()
                return

            else:
                return

        self.mark_missing()
        if self.owner not in [0, None]:
            self.try_capture(msg.poses)


    def select_facing(self, msg):
        for p in msg.poses:
            if self.is_facing(p):
                self.mark_target(p.id)


    def stop(self):
        goal = PoseStamped()

        goal.header.frame_id = self.pose_frame_id
        goal.header.stamp    = rospy.Time.now()
        goal.pose.position.x = 0.0
        goal.pose.position.y = 0.0
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)



    def follow(self, p):
        """Publishes basic_target from move_base."""
        dx = p.position.x
        dz = p.position.z

        d = math.hypot(dx, dz)
        if d < 1e-6:
            return

        goal_d = d - FOLLOW_STANDOFF

        if goal_d < 0:
            self.stop()
            return

        x = (dx / d) * goal_d
        z = (dz / d) * goal_d

        goal = PoseStamped()
        goal.header.frame_id = self.pose_frame_id
        goal.header.stamp    = rospy.Time.now()
        goal.pose.position.x = (dx / d) * goal_d
        goal.pose.position.y = 0.0
        goal.pose.position.z = (dz / d) * goal_d
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)



    def is_facing(self, pose):
        # Check both eyes are visible, and nose is roughly middle of ears
        left_eye, right_eye = pose.keypoints[1], pose.keypoints[2]
        nose, left_ear, right_ear = pose.keypoints[0], pose.keypoints[3], pose.keypoints[4]

        if left_eye.conf < VIS_THRESHOLD or right_eye.conf < VIS_THRESHOLD:
            return False
        return nose.x < left_ear.x and right_ear.x < nose.x




###################
### ENTRY POINT ###
###################
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_root = rospack.get_path("person_follower")
    node = FollowStateNode(StateMachine.from_json(package_root + "/scripts/states.json"))
    rospy.spin()