import math
import random
import time
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
GOAL_REACHED_DIST = 0.3
FORMATION_REACHED_DIST = 0.5
COLLISION_DIST = 0.2
COLLISION_HOLD_STEPS = 1
RESET_COLLISION_IGNORE_STEPS = 2
TIME_DELTA = 0.1
def check_pos(x, y):
    goal_ok = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False
    return goal_ok
class GazeboEnv:
    def __init__(
        self,
        launchfile,
        environment_dim,
        robot_names=None,
        *,
        include_other_poses=True,
        fixed_starts=None,
        fixed_goals=None,
        fixed_yaws=None,
    ):
        self.environment_dim = environment_dim
        self.robot_names = robot_names or ["p3dx"]
        self.num_robots = len(self.robot_names)
        self.include_other_poses = bool(include_other_poses)
        self.fixed_starts = fixed_starts
        self.fixed_goals = fixed_goals
        self.fixed_yaws = fixed_yaws
        self.upper, self.lower, self.actionfactor = 5.0, -5.0, 0.51

        self.speed_reward_scale = float(rospy.get_param("~speed_reward_scale", 0.2))
        self.gauss_gamma0 = float(rospy.get_param("~gauss_gamma0", -0.5))
        self.gauss_Gg = float(rospy.get_param("~gauss_Gg", 1.0))
        self.gauss_sigma_x = float(rospy.get_param("~gauss_sigma_x", 2.0))
        self.gauss_sigma_y = float(rospy.get_param("~gauss_sigma_y", 2.0))

        # Formation reward (followers): encourage staying near ideal relative positions to leader (robot 0).
        # Offsets are expressed in the leader's body frame (x forward, y left).
        self.enable_formation_reward = bool(rospy.get_param("~enable_formation_reward", True))
        self.formation_reward_k = float(rospy.get_param("~formation_reward_k", 1.0))
        self.formation_mode = str(rospy.get_param("~formation_mode", "triangle")).lower()
        self.enable_random_boxes = bool(rospy.get_param("~enable_random_boxes", False))
        default_tri = [-0.8, 0.6, -0.8, -0.6]  # follower1, follower2
        default_line = [-0.8, 0.0, -1.6, 0.0]
        self.triangle_offsets = list(rospy.get_param("~triangle_offsets", default_tri))
        self.line_offsets = list(rospy.get_param("~line_offsets", default_line))

        self.robots = []
        for n in self.robot_names:
            self.robots.append({
                "name": n,
                "odom_x": 0.0,
                "odom_y": 0.0,
                "odom_yaw": 0.0,
                "goal_x": 1.0,
                "goal_y": 0.0,
                "velodyne_data": np.ones(self.environment_dim) * 10.0,
                "last_odom": None,
                "prev_distance": None,
                "collision_count": 0,
                "reset_ignore_steps": 0,
                "terminated": False,
                "success": False,
            })

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim])
        self.gaps[-1][-1] += 0.03

        rospy.init_node("gym", anonymous=True)
        self.vel_pubs = []
        for i, n in enumerate(self.robot_names):
            self.vel_pubs.append(rospy.Publisher(f"/{n}/cmd_vel", Twist, queue_size=1))
            rospy.Subscriber(f"/{n}/velodyne_points", PointCloud2, self._mk_vel_cb(i), queue_size=1)
            rospy.Subscriber(f"/{n}/odom", Odometry, self._mk_odom_cb(i), queue_size=1)

        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=20)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.goal_pub = rospy.Publisher("goal_point", MarkerArray, queue_size=5)

    def _other_pose_features(self, self_idx):
        if not self.include_other_poses or self.num_robots <= 1:
            return np.zeros(0, dtype=np.float32)

        me = self.robots[self_idx]
        me_x, me_y = float(me["odom_x"]), float(me["odom_y"])
        me_yaw = float(me.get("odom_yaw", 0.0))
        feats = []
        for j, other in enumerate(self.robots):
            if j == self_idx:
                continue
            ox, oy = float(other["odom_x"]), float(other["odom_y"])
            dx = ox - me_x
            dy = oy - me_y
            dist = float(math.sqrt(dx * dx + dy * dy))
            bearing = float(math.atan2(dy, dx) - me_yaw)
            if bearing > math.pi:
                bearing -= 2.0 * math.pi
            if bearing < -math.pi:
                bearing += 2.0 * math.pi
            feats.extend([dist, bearing])
        return np.asarray(feats, dtype=np.float32)

    def _mk_vel_cb(self, i):
        def cb(v):
            data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
            out = np.ones(self.environment_dim) * 10.0
            for p in data:
                if p[2] <= -0.2:
                    continue
                mag = math.sqrt(p[0] * p[0] + p[1] * p[1])
                if mag < 1e-6:
                    continue
                beta = math.acos(p[0] / mag) * np.sign(p[1])
                dist = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                for j, g in enumerate(self.gaps):
                    if g[0] <= beta < g[1]:
                        out[j] = min(out[j], dist)
                        break
            self.robots[i]["velodyne_data"] = out
        return cb

    def _mk_odom_cb(self, i):
        def cb(o):
            self.robots[i]["last_odom"] = o
        return cb

    @staticmethod
    def _yaw_from_odom(o):
        q = o.pose.pose.orientation
        return float(Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2])

    def _refresh_odoms(self):
        for r in self.robots:
            o = r.get("last_odom", None)
            if o is None:
                continue
            r["odom_x"] = float(o.pose.pose.position.x)
            r["odom_y"] = float(o.pose.pose.position.y)
            r["odom_yaw"] = self._yaw_from_odom(o)

    def _formation_reward_and_error(self, follower_idx):
        if not self.enable_formation_reward:
            return 0.0, 0.0, (0.0, 0.0)
        if self.num_robots <= 1 or follower_idx == 0:
            return 0.0, 0.0, (0.0, 0.0)

        leader = self.robots[0]
        fol = self.robots[follower_idx]

        offsets = self.triangle_offsets if self.formation_mode.startswith("tri") else self.line_offsets
        expected_len = 2 * (self.num_robots - 1)
        if len(offsets) != expected_len:
            return 0.0, 0.0, (0.0, 0.0)

        # follower order excludes leader: follower_idx 1 -> offsets[0:2], follower_idx 2 -> offsets[2:4], ...
        k = follower_idx - 1
        dx_b = float(offsets[2 * k + 0])
        dy_b = float(offsets[2 * k + 1])

        cy = math.cos(float(leader["odom_yaw"]))
        sy = math.sin(float(leader["odom_yaw"]))
        dx_w = cy * dx_b - sy * dy_b
        dy_w = sy * dx_b + cy * dy_b

        ideal_x = float(leader["odom_x"]) + dx_w
        ideal_y = float(leader["odom_y"]) + dy_w

        ex = float(fol["odom_x"]) - ideal_x
        ey = float(fol["odom_y"]) - ideal_y
        err = float(math.sqrt(ex * ex + ey * ey))
        return -self.formation_reward_k * err, err, (ex, ey)

    def _single_state(self, i, r, action):
        if r["terminated"]:
            if r["last_odom"] is not None:
                r["odom_x"] = float(r["last_odom"].pose.pose.position.x)
                r["odom_y"] = float(r["last_odom"].pose.pose.position.y)
                r["odom_yaw"] = self._yaw_from_odom(r["last_odom"])
            dx = r["goal_x"] - r["odom_x"]
            dy = r["goal_y"] - r["odom_y"]
            distance = np.linalg.norm([dx, dy])
            state = np.concatenate([
                r["velodyne_data"],
                np.array([distance, 0.0, 0.0, 0.0], dtype=np.float32),
                self._other_pose_features(i),
            ])
            return state.astype(np.float32), 0.0, True, r["success"], False

        done, collision_detected, min_laser = self.observe_collision(r["velodyne_data"])
        if r["reset_ignore_steps"] > 0:
            r["reset_ignore_steps"] -= 1
            collision_detected = False
        if collision_detected:
            r["collision_count"] += 1
        else:
            r["collision_count"] = 0
        collision = r["collision_count"] >= COLLISION_HOLD_STEPS
        if r["last_odom"] is not None:
            r["odom_x"] = float(r["last_odom"].pose.pose.position.x)
            r["odom_y"] = float(r["last_odom"].pose.pose.position.y)
            r["odom_yaw"] = self._yaw_from_odom(r["last_odom"])
        angle = float(r.get("odom_yaw", 0.0))

        dx = r["goal_x"] - r["odom_x"]
        dy = r["goal_y"] - r["odom_y"]
        distance = np.linalg.norm([dx, dy])
        mag = math.sqrt(dx * dx + dy * dy)
        beta = 0.0 if mag < 1e-6 else math.acos(dx / mag)
        if dy < 0:
            beta = -beta
        theta = beta - angle
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        target = distance < GOAL_REACHED_DIST
        formation_ok = False
        if i != 0:
            _, ferr, _ = self._formation_reward_and_error(i)
            formation_ok = float(ferr) < float(FORMATION_REACHED_DIST)

        if i == 0:
            done = done or target
            if target:
                r["terminated"] = True
                r["success"] = True
            elif collision:
                r["terminated"] = True
                r["success"] = False
        else:
            # Followers do not terminate the episode by reaching their own goal.
            # Collision termination for followers is handled as a global condition in step().
            if collision:
                r["terminated"] = True
                r["success"] = False

        prev_distance = r["prev_distance"]
        r["prev_distance"] = distance

        state = np.concatenate([
            r["velodyne_data"],
            np.array([distance, theta, action[0], action[1]], dtype=np.float32),
            self._other_pose_features(i),
        ])
        reward = self.get_reward(target, collision, action, min_laser, dx, dy, self.actionfactor)
        if i != 0:
            reward = 0.0 if collision else float(action[0]) - 0.5 * abs(float(action[1])) * float(self.actionfactor)
            if not r["terminated"]:
                fr, _, _ = self._formation_reward_and_error(i)
                reward += float(fr)
        return state.astype(np.float32), reward, done, (target if i == 0 else formation_ok), collision

    def step(self, actions):
        for i, a in enumerate(actions):
            cmd = Twist()
            if self.robots[i]["terminated"]:
                cmd.linear.x, cmd.angular.z = 0.0, 0.0
            else:
                cmd.linear.x, cmd.angular.z = a[0], a[1]
            self.vel_pubs[i].publish(cmd)

        self.publish_markers()
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause()

        # Make sure all robots' odom_x/odom_y/odom_yaw are from the same sim tick
        self._refresh_odoms()

        states, rewards, dones, targets, collisions = [], [], [], [], []
        formation_errors = [0.0 for _ in range(self.num_robots)]
        formation_error_vecs = [(0.0, 0.0) for _ in range(self.num_robots)]
        was_active = [not r["terminated"] for r in self.robots]
        for i, r in enumerate(self.robots):
            s, rw, d, t, c = self._single_state(i, r, actions[i])
            states.append(s)
            rewards.append(rw)
            dones.append(d)
            targets.append(t)
            collisions.append(c)
            if i != 0:
                _, err, evec = self._formation_reward_and_error(i)
                formation_errors[i] = float(err)
                formation_error_vecs[i] = (float(evec[0]), float(evec[1]))

        active_rewards = [rw for rw, active in zip(rewards, was_active) if active]
        team_reward = float(np.mean(active_rewards)) if active_rewards else 0.0

        leader_target = bool(targets[0]) if len(targets) > 0 else False
        followers_formation_ok = all(bool(t) for t in targets[1:]) if self.num_robots > 1 else True
        team_success = bool(leader_target and followers_formation_ok)
        any_collision = any(bool(c) for c in collisions)
        team_done = bool(team_success or any_collision)
        dones = [team_done for _ in range(self.num_robots)]
        robot_poses = [
            (float(r.get("odom_x", 0.0)), float(r.get("odom_y", 0.0)), float(r.get("odom_yaw", 0.0)))
            for r in self.robots
        ]
        info = {
            "agent_rewards": rewards,
            "agent_dones": dones,
            "agent_targets": targets,
            "agent_collisions": collisions,
            "leader_target": leader_target,
            "followers_formation_ok": followers_formation_ok,
            "team_success": team_success,
            "termination_reason": "success" if team_success else ("collision" if any_collision else "running"),
            "formation_errors": formation_errors,
            "formation_error_vecs": formation_error_vecs,
            "robot_poses": robot_poses,
        }
        return np.array(states, dtype=np.float32), team_reward, team_done, info

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_proxy()

        occupied = []
        for idx, r in enumerate(self.robots):
            # Sample or fix start position
            if self.fixed_starts is not None:
                x, y = float(self.fixed_starts[idx][0]), float(self.fixed_starts[idx][1])
            else:
                x, y = self._sample_pos(occupied, 1.2)

            # Initial yaw priority:
            # 1) fixed_yaws (if provided)
            # 2) point from start to goal (when goals are fixed)
            # 3) random yaw
            if self.fixed_yaws is not None:
                angle = float(self.fixed_yaws[idx])
            elif self.fixed_goals is not None:
                gx, gy = float(self.fixed_goals[idx][0]), float(self.fixed_goals[idx][1])
                angle = float(math.atan2(gy - y, gx - x))
            else:
                angle = float(np.random.uniform(-np.pi, np.pi))

            q = Quaternion.from_euler(0.0, 0.0, angle)
            occupied.append((x, y))
            m = ModelState()
            m.model_name = r["name"]
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.01
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q.x, q.y, q.z, q.w
            self.set_state.publish(m)
            r["odom_x"], r["odom_y"] = x, y
            r["odom_yaw"] = float(angle)
            r["terminated"] = False
            r["success"] = False
            r["collision_count"] = 0
            r["reset_ignore_steps"] = RESET_COLLISION_IGNORE_STEPS
            r["prev_distance"] = None

        for idx, r in enumerate(self.robots):
            if self.fixed_goals is not None:
                if idx == 0:
                    r["goal_x"], r["goal_y"] = float(self.fixed_goals[idx][0]), float(self.fixed_goals[idx][1])
                else:
                    # Followers share the leader goal so their state distance/theta remains defined,
                    # but episode termination does not depend on follower goal reaching.
                    r["goal_x"], r["goal_y"] = float(self.fixed_goals[0][0]), float(self.fixed_goals[0][1])
            else:
                # Sample only leader goal; followers share the leader goal.
                if idx == 0:
                    self.change_goal(r)
                else:
                    r["goal_x"], r["goal_y"] = float(self.robots[0]["goal_x"]), float(self.robots[0]["goal_y"])
        if self.enable_random_boxes:
            self.random_box()
        self.publish_markers()

        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause()

        self._refresh_odoms()
        states = []
        for i, r in enumerate(self.robots):
            s, _, _, _, _ = self._single_state(i, r, [0.0, 0.0])
            states.append(s)
        return np.array(states, dtype=np.float32)

    def _sample_pos(self, occupied, min_dist):
        while True:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            if not check_pos(x, y):
                continue
            if all(np.linalg.norm([x - ox, y - oy]) >= min_dist for ox, oy in occupied):
                return x, y

    def change_goal(self, r):
        self.upper = min(10, self.upper + 0.004)
        self.lower = max(-10, self.lower - 0.004)
        self.actionfactor = max(0.0, self.actionfactor - 0.00005)
        while True:
            gx = r["odom_x"] + random.uniform(self.upper, self.lower)
            gy = r["odom_y"] + random.uniform(self.upper, self.lower)
            if check_pos(gx, gy):
                r["goal_x"], r["goal_y"] = gx, gy
                return

    def random_box(self):
        avoid = [(r["odom_x"], r["odom_y"]) for r in self.robots] + [(r["goal_x"], r["goal_y"]) for r in self.robots]
        for i in range(4):
            while True:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                if not check_pos(x, y):
                    continue
                if all(np.linalg.norm([x - px, y - py]) >= 1.5 for px, py in avoid):
                    break
            m = ModelState()
            m.model_name = f"cardboard_box_{i}"
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.0
            m.pose.orientation.w = 1.0
            self.set_state.publish(m)

    def publish_markers(self):
        arr = MarkerArray()
        for i, r in enumerate(self.robots):
            m = Marker()
            m.header.frame_id = "world"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.scale.x = m.scale.y = 0.15
            m.scale.z = 0.02
            m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 0.1, 0.9, 0.1
            m.pose.orientation.w = 1.0
            m.pose.position.x, m.pose.position.y = r["goal_x"], r["goal_y"]
            arr.markers.append(m)
        self.goal_pub.publish(arr)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    def get_reward(self, target, collision, action, min_laser, dx, dy, actionfactor):
        if target:
            return 100.0
        if collision:
            return -100.0

        sx2 = max(1e-6, float(self.gauss_sigma_x) ** 2)
        sy2 = max(1e-6, float(self.gauss_sigma_y) ** 2)
        gauss_reward = float(self.gauss_gamma0) + float(self.gauss_Gg) * math.exp(
            -(((dx * dx) / sx2) + ((dy * dy) / sy2))
        )

        forward_reward = action[0]
        turn_penalty = -0.5 * abs(action[1]) * float(actionfactor)
        clearance_penalty = -0.5 * max(0.0, 1.2 - min_laser)
        return forward_reward + turn_penalty + clearance_penalty + gauss_reward
