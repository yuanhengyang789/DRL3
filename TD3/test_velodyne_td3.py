from __future__ import annotations
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from velodyne_env import GazeboEnv

def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _leader_offset_target_xy(leader_pose, offset_body_xy):
    lx, ly, lyaw = float(leader_pose[0]), float(leader_pose[1]), float(leader_pose[2])
    dx_b, dy_b = float(offset_body_xy[0]), float(offset_body_xy[1])
    c = math.cos(lyaw)
    s = math.sin(lyaw)
    tx = lx + c * dx_b - s * dy_b
    ty = ly + s * dx_b + c * dy_b
    return float(tx), float(ty)

def _parse_pose_text(text):
    vals = [float(v) for v in (text or "").split()]
    vals += [0.0] * (6 - len(vals))
    return vals[:6]

def _world_pose(parent_pose, child_pose):
    px, py, pz, pr, pp, pyaw = parent_pose
    cx, cy, cz, cr, cp, cyaw = child_pose
    c = math.cos(pyaw)
    s = math.sin(pyaw)
    return [
        px + c * cx - s * cy,
        py + s * cx + c * cy,
        pz + cz,
        pr + cr,
        pp + cp,
        pyaw + cyaw,
    ]

def _extract_td32_map(world_path):
    static_obstacles = []
    dynamic_paths = []
    tree = ET.parse(world_path)
    root = tree.getroot()
    for model in root.findall(".//model"):
        model_name = model.get("name", "")
        is_static = model.findtext("static") == "true"
        model_pose = _parse_pose_text(model.findtext("pose"))

        for plugin in model.findall("plugin"):
            start_text = plugin.findtext("start")
            end_text = plugin.findtext("end")
            if start_text is None or end_text is None:
                continue
            start = _parse_pose_text(start_text)
            end = _parse_pose_text(end_text)
            dynamic_paths.append({
                "name": model_name,
                "start": (float(start[0]), float(start[1])),
                "end": (float(end[0]), float(end[1])),
            })

        if not is_static:
            continue

        for link in model.findall("link"):
            link_pose = _world_pose(model_pose, _parse_pose_text(link.findtext("pose")))
            for collision in link.findall("collision"):
                box_size = collision.findtext("geometry/box/size")
                if box_size is None:
                    continue
                sx, sy, sz = [float(v) for v in box_size.split()[:3]]
                if sx <= 0.0 or sy <= 0.0:
                    continue
                collision_pose = _world_pose(link_pose, _parse_pose_text(collision.findtext("pose")))
                x, y, _, _, _, yaw = collision_pose
                static_obstacles.append({
                    "name": model_name,
                    "x": float(x),
                    "y": float(y),
                    "width": float(sx),
                    "height": float(sy),
                    "yaw": float(yaw),
                    "z": float(collision_pose[2]),
                    "size_z": float(sz),
                })
    return static_obstacles, dynamic_paths

def _plot_episode_path(traj, starts, goal, world_path, output_path):
    static_obstacles, dynamic_paths = _extract_td32_map(world_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("TD32 three-robot path planning result")
    ax.grid(True, linestyle="--", alpha=0.25)

    ax.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], color="black", linewidth=1.6, label="boundary")

    for obs in static_obstacles:
        name = obs["name"]
        if name in {"ground_plane", "fence"}:
            continue
        rect = patches.Rectangle(
            (obs["x"] - obs["width"] / 2.0, obs["y"] - obs["height"] / 2.0),
            obs["width"],
            obs["height"],
            facecolor="#BDBDBD",
            edgecolor="#424242",
            alpha=0.70,
            linewidth=1.0,
            label="static obstacle",
        )
        rect.set_transform(
            Affine2D().rotate_around(float(obs["x"]), float(obs["y"]), float(obs["yaw"])) + ax.transData
        )
        ax.add_patch(rect)

    for idx, path in enumerate(dynamic_paths, start=1):
        sx, sy = path["start"]
        ex, ey = path["end"]
        ax.plot([sx, ex], [sy, ey], color="black", linestyle="--", linewidth=2.0, label="dynamic obstacle path")
        ax.scatter(sx, sy, color="black", marker="^", s=70, zorder=6)
        ax.scatter(ex, ey, color="black", marker="v", s=70, zorder=6)
        ax.text(sx + 0.08, sy + 0.08, f"D{idx}S", color="black", weight="bold")
        ax.text(ex + 0.08, ey + 0.08, f"D{idx}E", color="black", weight="bold")

    colors = ["#D62728", "#1F77B4", "#2CA02C"]
    labels = ["leader", "follower 1", "follower 2"]
    for i, points in enumerate(traj):
        arr = np.asarray(points, dtype=np.float32)
        if len(arr) > 0:
            ax.plot(arr[:, 0], arr[:, 1], color=colors[i], linewidth=2.1, label=labels[i])
        ax.scatter(starts[i][0], starts[i][1], color=colors[i], marker="o", s=75, edgecolor="black", zorder=7)
        ax.text(starts[i][0] + 0.08, starts[i][1] + 0.08, f"S{i}", color=colors[i], weight="bold")

    ax.scatter(goal[0], goal[1], color="gold", marker="*", s=190, edgecolor="black", zorder=8, label="leader goal")
    ax.text(goal[0] + 0.08, goal[1] + 0.08, "G", color="#8A6D00", weight="bold")

    handles, labels_seen = ax.get_legend_handles_labels()
    unique = dict(zip(labels_seen, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

class RelativeFormationFollower:
    V_MAX = 0.95
    W_MAX = 1.2
    DV_MAX = 0.2
    DW_MAX = 0.5
    SAFETY_DV_MAX = 0.85
    SAFETY_DW_MAX = 1.45

    DIST_ON = 0.14
    DIST_OFF = 0.06
    V_BASE_GAIN = 0.40
    EX_GAIN = 0.70

    LAMBDA_Y = 1.05
    LAMBDA_YAW = 0.90
    K_EQ_Y = 1.70
    K_EQ_YAW = 1.35
    ETA_Y = 1.25
    ETA_YAW = 1.05
    PHI_Y = 0.12
    PHI_YAW = 0.22
    HEADING_GAIN = 1.35

    AVOID_DIST = 0.75
    SLOW_DIST = 0.6
    STOP_DIST = 0.35
    EMERGENCY_DIST = 0.3
    CLEAR_DIST = 0.62
    AVOID_W_GAIN = 1.45
    EMERGENCY_W_GAIN = 1.30
    AVOID_PHI = 0.85
    ETA_REP = 1.08

    def __init__(self):
        self.prev_cmd = np.array([0.0, 0.0], dtype=np.float32)
        self.last_debug = {}

    def reset(self):
        self.prev_cmd[:] = 0.0
        self.last_debug = {}

    @staticmethod
    def _sat(value, width):
        width = max(float(width), 1e-6)
        return float(np.clip(value / width, -1.0, 1.0))

    def _repulsive_avoidance(self, laser_ranges):
        if laser_ranges is None or len(laser_ranges) == 0:
            return {
                "scale": 1.0,
                "turn": 0.0,
                "min_laser": 10.0,
                "rep_x": 0.0,
                "rep_y": 0.0,
                "front": 10.0,
                "left": 10.0,
                "right": 10.0,
                "active": False,
                "emergency": False,
            }

        n = len(laser_ranges)
        start_ang = -math.pi / 2.0
        step = math.pi / max(1, n)
        rep_x = 0.0
        rep_y = 0.0
        min_range = 10.0
        front_risk = 0.0
        left_risk = 0.0
        right_risk = 0.0
        front_clearance = 10.0
        left_clearance = 10.0
        right_clearance = 10.0

        for i, raw_d in enumerate(laser_ranges):
            d = float(raw_d)
            if not math.isfinite(d) or d <= 0.0:
                continue

            ang = start_ang + (i + 0.5) * step
            abs_ang = abs(ang)
            min_range = min(min_range, d)

            front_weight = max(0.0, math.cos(ang)) ** 2
            side_weight = max(0.0, math.sin(abs_ang)) ** 1.4
            risk = max(0.0, (self.AVOID_DIST - d) / max(self.AVOID_DIST, 1e-6))

            if front_weight > 0.10:
                front_clearance = min(front_clearance, d)
            if ang > 0.0:
                left_clearance = min(left_clearance, d)
                left_risk = max(left_risk, risk * side_weight)
            elif ang < 0.0:
                right_clearance = min(right_clearance, d)
                right_risk = max(right_risk, risk * side_weight)
            front_risk = max(front_risk, risk * front_weight)

            if d >= self.AVOID_DIST:
                continue

            strength = self.ETA_REP * (0.18 + 0.82 * front_weight) * (1.0 / max(d, 1e-3) - 1.0 / self.AVOID_DIST)
            rep_x += -strength * math.cos(ang)
            rep_y += -strength * math.sin(ang)

        if front_clearance < self.STOP_DIST:
            slow_scale = 0.0
        elif front_clearance < self.SLOW_DIST:
            ratio = (front_clearance - self.STOP_DIST) / max(1e-6, self.SLOW_DIST - self.STOP_DIST)
            slow_scale = 0.10 + 0.65 * ratio
        elif front_risk > 0.0:
            slow_scale = 1.0 - 0.35 * front_risk
        else:
            slow_scale = 1.0

        avoid_heading = math.atan2(rep_y, rep_x) if abs(rep_x) + abs(rep_y) > 1e-6 else 0.0
        avoid_turn = -self.AVOID_W_GAIN * self._sat(avoid_heading, self.AVOID_PHI)
        side_bias = right_risk - left_risk
        avoid_turn += 0.45 * self.AVOID_W_GAIN * float(np.clip(side_bias, -1.0, 1.0))

        active = front_risk > 0.08 or max(left_risk, right_risk) > 0.45
        emergency = front_clearance < self.STOP_DIST or min_range < self.EMERGENCY_DIST

        if emergency:
            turn_to_clear_side = -1.0 if left_clearance > right_clearance else 1.0
            avoid_turn = self.W_MAX * turn_to_clear_side
            slow_scale = 0.0

        return {
            "scale": float(np.clip(slow_scale, 0.0, 1.0)),
            "turn": float(np.clip(avoid_turn, -self.W_MAX, self.W_MAX)),
            "min_laser": float(min_range),
            "rep_x": float(rep_x),
            "rep_y": float(rep_y),
            "front": float(front_clearance),
            "left": float(left_clearance),
            "right": float(right_clearance),
            "active": bool(active),
            "emergency": bool(emergency),
        }

    def act(self, leader_pose, follower_pose, offset_body_xy, leader_cmd, laser_ranges=None):
        lx, ly, lyaw = float(leader_pose[0]), float(leader_pose[1]), float(leader_pose[2])
        fx, fy, fyaw = float(follower_pose[0]), float(follower_pose[1]), float(follower_pose[2])
        dx_d, dy_d = float(offset_body_xy[0]), float(offset_body_xy[1])
        leader_v, leader_w = float(leader_cmd[0]), float(leader_cmd[1])

        rel_x_w = fx - lx
        rel_y_w = fy - ly
        c = math.cos(lyaw)
        s = math.sin(lyaw)

        rel_x_b = c * rel_x_w + s * rel_y_w
        rel_y_b = -s * rel_x_w + c * rel_y_w

        e_x = rel_x_b - dx_d
        e_y = rel_y_b - dy_d

        target_x, target_y = _leader_offset_target_xy(leader_pose, offset_body_xy)
        dist_err = math.hypot(e_x, e_y)

        # In this system, positive angular command makes yaw decrease.
        # So the control-oriented yaw errors are defined as current - desired.
        e_yaw = _wrap_pi(fyaw - lyaw)
        if dist_err < 0.08:
            heading_to_target = 0.0
        else:
            desired_heading = math.atan2(target_y - fy, target_x - fx)
            heading_to_target = _wrap_pi(fyaw - desired_heading)

        s_y = e_y + self.LAMBDA_Y * e_yaw
        s_yaw = e_yaw

        v_cmd = leader_v + self.V_BASE_GAIN * leader_v + self.EX_GAIN * max(0.0, -e_x)

        heading_scale = max(0.35, math.cos(heading_to_target))
        yaw_scale = max(0.40, math.cos(e_yaw))
        v_cmd *= heading_scale * yaw_scale

        if abs(heading_to_target) > 1.20:
            v_cmd *= 0.50
        elif abs(heading_to_target) > 0.80 or abs(e_yaw) > 0.90:
            v_cmd *= 0.72

        w_eq = (
            leader_w
            + self.K_EQ_Y * e_y
            + self.K_EQ_YAW * math.sin(e_yaw)
            + self.HEADING_GAIN * heading_to_target
        )
        w_sw = self.ETA_Y * self._sat(s_y, self.PHI_Y) + self.ETA_YAW * self._sat(s_yaw, self.PHI_YAW)
        w_cmd = w_eq + w_sw

        if dist_err < 0.10 and abs(e_yaw) < 0.20:
            v_cmd = min(v_cmd, leader_v)
            w_cmd = 0.5 * leader_w

        avoid = self._repulsive_avoidance(laser_ranges)
        avoid_scale = avoid["scale"]
        avoid_turn = avoid["turn"]
        min_laser = avoid["min_laser"]
        rep_x = avoid["rep_x"]
        rep_y = avoid["rep_y"]

        if avoid["emergency"]:
            v_cmd = 0.0
            w_cmd = avoid_turn
        elif avoid["active"]:
            v_cmd *= avoid_scale
            w_cmd = 0.30 * w_cmd +  avoid_turn
        else:
            v_cmd *= avoid_scale

        v_cmd = float(np.clip(v_cmd, 0.0, self.V_MAX))
        w_cmd = float(np.clip(w_cmd, -self.W_MAX, self.W_MAX))

        prev_v, prev_w = float(self.prev_cmd[0]), float(self.prev_cmd[1])
        dv_max = self.SAFETY_DV_MAX if avoid["active"] else self.DV_MAX
        dw_max = self.SAFETY_DW_MAX if avoid["active"] else self.DW_MAX
        v_cmd = float(np.clip(v_cmd, prev_v - dv_max, prev_v + dv_max))
        w_cmd = float(np.clip(w_cmd, prev_w - dw_max, prev_w + dw_max))
        self.prev_cmd[:] = [v_cmd, w_cmd]
        self.last_debug = {
            "leader_pose": (lx, ly, lyaw),
            "follower_pose": (fx, fy, fyaw),
            "target_xy": (target_x, target_y),
            "offset": (dx_d, dy_d),
            "leader_cmd": (leader_v, leader_w),
            "errors": {
                "e_x": float(e_x),
                "e_y": float(e_y),
                "e_yaw": float(e_yaw),
                "dist_err": float(dist_err),
                "heading_to_target": float(heading_to_target),
                "s_y": float(s_y),
                "s_yaw": float(s_yaw),
            },
            "avoid": {
                "scale": float(avoid_scale),
                "turn": float(avoid_turn),
                "min_laser": float(min_laser),
                "front": float(avoid["front"]),
                "left": float(avoid["left"]),
                "right": float(avoid["right"]),
                "active": bool(avoid["active"]),
                "emergency": bool(avoid["emergency"]),
                "rep_x": float(rep_x),
                "rep_y": float(rep_y),
            },
            "cmd": {
                "v_cmd": float(v_cmd),
                "w_cmd": float(w_cmd),
                "w_eq": float(w_eq),
                "w_sw": float(w_sw),
            },
        }
        return self.prev_cmd.copy()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        return self.tanh(self.layer_3(s))

class LeaderPolicy:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        return self.actor(s).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=device)
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 300
file_name = "TD3_velodyne"
environment_dim = 20
robot_dim = 4
action_dim = 2
num_agents = 3
robot_names = ["p3dx_0", "p3dx_1", "p3dx_2"]
leader_idx = 0
local_state_dim = environment_dim + robot_dim
use_fixed_start_goal = True
fixed_goals = [(4.0, 4.0), (4.0, 2.6), (2.6, 4.0)]
fixed_leader_start = (-3.0, -3.0)
LEADER_V_MAX = 0.5
follower_offsets = [(-0.8, 0.6), (-0.8, -0.6)]
DEBUG_FOLLOWERS = True
DEBUG_PRINT_EVERY = 10
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORLD_PATH = PROJECT_ROOT / "catkin_ws" / "src" / "multi_robot_scenario" / "launch" / "TD32.world"
PLOT_OUTPUT_PATH = SCRIPT_DIR / "results" / "TD32_episode_path.png"

def _formation_fixed_starts(leader_start, leader_goal, offsets):
    lx, ly = float(leader_start[0]), float(leader_start[1])
    gx, gy = float(leader_goal[0]), float(leader_goal[1])
    yaw = math.atan2(gy - ly, gx - lx)
    c, s = math.cos(yaw), math.sin(yaw)
    starts = [(lx, ly)]
    for dx_b, dy_b in offsets:
        starts.append((lx + c * dx_b - s * dy_b, ly + s * dx_b + c * dy_b))
    return starts

fixed_starts = _formation_fixed_starts(fixed_leader_start, fixed_goals[0], follower_offsets)

env = GazeboEnv(
    "multi_robot_scenario.launch",
    environment_dim,
    robot_names=robot_names,
    include_other_poses=False,
    fixed_starts=fixed_starts if use_fixed_start_goal else None,
    fixed_goals=fixed_goals if use_fixed_start_goal else None,
)

time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

policy = LeaderPolicy(local_state_dim, action_dim)
try:
    policy.load(file_name, "./models")
except Exception as e:
    raise ValueError(f"Could not load model: {e}")

followers = [RelativeFormationFollower(), RelativeFormationFollower()]

done = False
episode_timesteps = 0
state = env.reset()
trajectories = [[(float(r["odom_x"]), float(r["odom_y"]))] for r in env.robots]

# 用于保存两个跟随车的误差数据（分别存储）
errors_follower1 = []   # 列表元素: (step, e_x, e_y, e_yaw)
errors_follower2 = []

while not done:
    state_np = np.asarray(state, dtype=np.float32)
    s_leader = state_np[leader_idx]
    leader_action_raw = np.clip(policy.get_action(s_leader).astype(np.float32), -1.0, 1.0)

    leader_v = float(np.clip((leader_action_raw[0] + 1.0) / 2.0, 0.0, LEADER_V_MAX))
    leader_w = float(0.55 * leader_action_raw[1])

    env_action = np.zeros((num_agents, action_dim), dtype=np.float32)
    env_action[leader_idx] = [leader_v, leader_w]

    lr = env.robots[leader_idx]
    leader_pose = (float(lr["odom_x"]), float(lr["odom_y"]), float(lr["odom_yaw"]))

    for i in range(1, num_agents):
        r = env.robots[i]
        follower_pose = (float(r["odom_x"]), float(r["odom_y"]), float(r["odom_yaw"]))
        env_action[i] = followers[i - 1].act(
            leader_pose,
            follower_pose,
            follower_offsets[i - 1],
            (leader_v, leader_w),
            r["velodyne_data"],
        )

    # 收集误差数据（分别存入两个列表）
    for idx, follower in enumerate(followers, start=1):
        dbg = follower.last_debug
        if dbg:
            err = dbg["errors"]
            if idx == 1:
                errors_follower1.append((episode_timesteps, err["e_x"], err["e_y"], err["e_yaw"]))
            else:  # idx == 2
                errors_follower2.append((episode_timesteps, err["e_x"], err["e_y"], err["e_yaw"]))

    if DEBUG_FOLLOWERS and episode_timesteps % DEBUG_PRINT_EVERY == 0:
        print(
            f"[step {episode_timesteps:03d}] leader_pose=({leader_pose[0]:.2f}, {leader_pose[1]:.2f}, {leader_pose[2]:.2f}) "
            f"leader_cmd=({leader_v:.2f}, {leader_w:.2f})"
        )
        for idx, follower in enumerate(followers, start=1):
            dbg = follower.last_debug
            if not dbg:
                continue
            err = dbg["errors"]
            cmd = dbg["cmd"]
            target_xy = dbg["target_xy"]
            fp = dbg["follower_pose"]
            avoid = dbg["avoid"]
            print(
                f"  F{idx} pose=({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f}) "
                f"target=({target_xy[0]:.2f}, {target_xy[1]:.2f}) "
                f"err[x={err['e_x']:.3f}, y={err['e_y']:.3f}, yaw={err['e_yaw']:.3f}, d={err['dist_err']:.3f}, h={err['heading_to_target']:.3f}] "
                f"avoid[min={avoid['min_laser']:.3f}, front={avoid['front']:.3f}, L={avoid['left']:.3f}, R={avoid['right']:.3f}, "
                f"scale={avoid['scale']:.2f}, turn={avoid['turn']:.3f}, active={avoid['active']}, emergency={avoid['emergency']}] "
                f"cmd[v={cmd['v_cmd']:.3f}, w={cmd['w_cmd']:.3f}]"
            )

    next_state, reward, done, info = env.step(env_action)
    for robot_idx, pose in enumerate(info.get("robot_poses", [])):
        trajectories[robot_idx].append((float(pose[0]), float(pose[1])))
    done = bool(done or episode_timesteps + 1 == max_ep)

    if done:
        ferr = [round(e, 3) for e in info["formation_errors"]]
        print(
            f"Episode end | steps={episode_timesteps + 1} | reward={reward:.3f}"
            f" | targets={info['agent_targets']} | collisions={info['agent_collisions']}"
            f" | formation_err={ferr}"
        )
        _plot_episode_path(
            trajectories,
            fixed_starts,
            fixed_goals[leader_idx],
            WORLD_PATH,
            PLOT_OUTPUT_PATH,
        )
        print(f"Path plot saved to: {PLOT_OUTPUT_PATH}")

        # 保存两个跟随车的误差数据为CSV文件
        import csv
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # 跟随车1
        csv_path1 = results_dir / "follower1_errors.csv"
        with open(csv_path1, 'w', newline='') as f1:
            writer = csv.writer(f1)
            writer.writerow(["step", "e_x", "e_y", "e_yaw"])
            writer.writerows(errors_follower1)
        print(f"Follower 1 errors saved to: {csv_path1}")

        # 跟随车2
        csv_path2 = results_dir / "follower2_errors.csv"
        with open(csv_path2, 'w', newline='') as f2:
            writer = csv.writer(f2)
            writer.writerow(["step", "e_x", "e_y", "e_yaw"])
            writer.writerows(errors_follower2)
        print(f"Follower 2 errors saved to: {csv_path2}")

        break

    state = next_state
    episode_timesteps += 1