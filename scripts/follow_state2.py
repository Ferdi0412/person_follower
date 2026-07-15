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

import actionlib
import tf2_ros
import tf2_geometry_msgs
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

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

################
### ROS NODE ###
################
class FollowStateNode:
    def __init__(self, state_machine):
        rospy.init_node("person_follower_state_node2")
        rospy.on_shutdown(self.release_target)

        self.state_machine = state_machine

        self.owner = None
        self.last_seen = None
        self.last_published = None

        self.pose_frame_id = None
        self.last_x = 0
        self.last_z = 0

        self.state_pub  = rospy.Publisher("/person_follower/state",  String, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher("/person_follower/target", UInt32, queue_size=1, latch=True)

        self.gesture_sub = rospy.Subscriber("/person_follower/gesture", String,    self.gesture_callback, queue_size=5)
        self.pose_sub    = rospy.Subscriber("/person_follower/pose",    PoseArray, self.pose_callback,    queue_size=1)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.client.wait_for_server()

        self.last_time = rospy.Time(0)
        self.interval = rospy.Duration(1.0)
        rospy.loginfo("person_follower_state_node2: starting...")


    def release_target(self):
        self.target_pub.publish(UInt32(data=0))
        self.stop()



    def gesture_callback(self, msg):
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
        self.target_pub.publish(UInt32(data=self.owner))
        self.pose_frame_id = msg.header.frame_id

        if self.owner in [None, 0]:
            self.select_facing(msg)
            return

        for p in msg.poses:
            if p.id != self.owner:
                continue

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



    def select_facing(self, msg):
        for p in msg.poses:
            if self.is_facing(p):
                self.owner = p.id
                self.target_pub.publish(UInt32(data=p.id))
                self.last_seen = 0
                return


    def stop(self):
        self.client.cancel_all_goals()


    def follow(self, p):
        """Publishes basic_target from move_base."""
        dx = p.position.x
        dz = p.position.z

        d = math.hypot(dx, dz)
        if d < 1e-6:
            return

        goal_d = d - FOLLOW_STANDOFF

        if goal_d < 0:
            return

        x = (dx / d) * goal_d
        z = (dz / d) * goal_d

        now = rospy.Time.now()
        if now - self.last_time < self.interval:
            return

        local_pose = PoseStamped()
        local_pose.header.frame_id = self.pose_frame_id
        local_pose.header.stamp    = rospy.Time(0)
        local_pose.pose.position.x = (dx / d) * goal_d
        local_pose.pose.position.y = 0.0
        local_pose.pose.position.z = (dz / d) * goal_d
        local_pose.pose.orientation.w = 1.0

        try:
            map_pose = self.tf_buffer.transform(local_pose, "map", timeout=rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(5.0, "person_follower_state_node2: couldn't apply tf transform - %s", e)
            return

        try:
            robot_pose = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), timeout=rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(5.0, "person_follower_state_node: robot pose lookup failed: %s", e)
            return

        rx = robot_pose.transform.translation.x
        ry = robot_pose.transform.translation.y
        gx = map_pose.pose.position.x
        gy = map_pose.pose.position.y
        yaw = math.atan2(gy - ry, gx - rx)

        qz = math.cos(yaw / 2.0)
        qw = math.sin(yaw / 2.0)
        map_pose.pose.orientation.x = 0.0
        map_pose.pose.orientation.y = 0.0
        map_pose.pose.orientation.z = qz
        map_pose.pose.orientation.w = qw

        goal = MoveBaseGoal()
        goal.target_pose = map_pose
        self.client.send_goal(goal)

        print("Sent target", map_pose, robot_pose)

        self.last_time = now


    def is_facing(self, pose):
        # Check both eyes are visible, and nose is roughly middle of ears
        left_eye, right_eye = pose.keypoints[1], pose.keypoints[2]
        nose, left_ear, right_ear = pose.keypoints[0], pose.keypoints[3], pose.keypoints[4]

        if left_eye.conf < VIS_THRESHOLD or right_eye.conf < VIS_THRESHOLD:
            return False
        return nose.x < left_ear.x and right_ear.x < nose.x



    def mark_missing(self):
        self.last_seen += 1
        if self.last_seen > KEEP_ALIVE:
            self.owner = None
            self.last_seen = None
            self.target_pub.publish(UInt32(data=0))



    def clear_all(self):
        self.target_pub.publish(UInt32(data=0))
        self.state_pub.publish("")



###################
### ENTRY POINT ###
###################
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_root = rospack.get_path("person_follower")
    node = FollowStateNode(StateMachine.from_json(package_root + "/scripts/states.json"))
    rospy.spin()