import warnings

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from collections import deque
from gymnasium.wrappers import TimeLimit
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class MyWalkerEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "walker2d_v5.xml",
        frame_skip: int = 4,
        default_camera_config: dict[str, float | np.ndarray] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.8, 2.0),
        healthy_angle_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        # task
        target_speed: float | None = None, 
        track_sigma: float = 0.5,
        speed_range: tuple[float, float] | None = None,  
        # symmestry
        w_sym: float = 0.0,          
        sym_k: float = 5.0,
        sym_window: int = 250,      # window size for symmetry evaluation
        sym_normalize: bool = False,  # divide the error by the joint-angle variance
        sym_eps: float = 0.005,
        sym_min_half: int = 1,       
        sym_amp_ratio: bool = False,  # multiply SYM by min/max of left/right joint variance 
        # touchdown detection
        min_air_steps: int = 5,     # minimum consecutive air steps required for a touchdown to be considered valid
        # flight
        w_flight: float = 0.0,
        flight_force_ratio: float = 0.1,
        # height
        w_height: float = 0.0,         # foot-clearance reward weight
        height_target: float = 0.08,   
        # stride
        w_stride: float = 0.0,       
        stride_min: float = 0.8,     # s; same-foot touchdown-to-touchdown time below this is penalized
        # cross
        w_cross: float = 0.0,        # penalty per touchdown whose step length is <= 0 (foot did not pass the other foot)
        # tsym
        w_tsym: float = 0.0,       
        tsym_sigma: float = 1.0,
        positive_reward: bool = False,  
        sym_kernel: str = "exp",
        curriculum_penalties_only: bool = False,  
        #step length
        step_len_min: float = 0.0,             
        step_len_speed_exp: float = 0.0,  
        step_len_v_ref: float = 1.0,
        #airtime
        w_airtime: float = 0.0, 
        airtime_threshold: float = 0.4, 
        airtime_cmd_min: float = 0.1,
        airtime_cutoff: bool = False,
        #place
        w_place: float = 0.0,      
        place_min: float = 0.15,    
        #footflat
        w_footflat: float = 0.0,         
        footflat_deadband: float = 10.0, 
        w_jlim: float = 0.0,      
        jlim_soft: float = 0.9,    
        jlim_joints: tuple = (2, 5), 
        stand_prob=0.0,     
        gait_gate_v=0.1,   
        #stand 
        w_stand=0.0,  
        cmd_dead_zone=0.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            target_speed,
            track_sigma,
            speed_range,
            w_sym,
            sym_k,
            sym_window,
            sym_normalize,
            sym_eps,
            sym_min_half,
            sym_amp_ratio,
            min_air_steps,
            w_flight,
            flight_force_ratio,
            w_height,
            height_target,
            w_stride,
            stride_min,
            w_cross,
            w_tsym,
            tsym_sigma,
            positive_reward,
            sym_kernel,
            curriculum_penalties_only,
            step_len_min,
            step_len_speed_exp,
            step_len_v_ref,
            w_airtime,
            airtime_threshold,
            airtime_cmd_min,
            airtime_cutoff,
            w_place,
            place_min,
            w_footflat,
            footflat_deadband,
            w_jlim,
            jlim_soft,
            jlim_joints,
            stand_prob,
            gait_gate_v,
            w_stand,
            cmd_dead_zone,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._target_speed = target_speed
        self._track_sigma = track_sigma
        self._speed_range = speed_range
        if speed_range is not None:
            self._target_speed = float(np.mean(speed_range))   # placeholder; resampled in reset_model

        self._w_sym = w_sym
        self._sym_k = sym_k
        self._joint_history = deque(maxlen=sym_window)
        self._sym_normalize = sym_normalize
        self._sym_eps = sym_eps
        self._sym_min_half = sym_min_half
        self._sym_amp_ratio = sym_amp_ratio
        self._sym_error = None
        self._step_count = 0

        self._min_air_steps = min_air_steps
        self._in_contact = [True, True]     # [right, left] whether each foot was in contact in the last step
        self._air_steps = [0, 0]            # [right, left] consecutive air steps
        self._contact_steps = [0, 0] 
        self._single_steps = [0, 0]         # [right, left] air steps counted only while the other foot is down
        self._swing_h = [0.0, 0.0]          # [right, left] max foot clearance in the current swing
        self._last_touchdown = None        
        self._last_step = None              
        self._recent_step_times = deque(maxlen=2)
        self._td_steps = {"R": deque(maxlen=3), "L": deque(maxlen=3)}   # step index
        self._touchdown = {}

        self._w_flight = w_flight
        self._flight_force_ratio = flight_force_ratio
        self._w_height = w_height
        self._height_target = height_target
        self._w_stride = w_stride
        self._stride_min = stride_min
        self._w_cross = w_cross
        self._w_tsym = w_tsym
        self._tsym_sigma = tsym_sigma
        self._style_scale = 1.0  
        self._curriculum_penalties_only = curriculum_penalties_only
        self._w_airtime = w_airtime
        self._airtime_threshold = airtime_threshold
        self._airtime_cmd_min = airtime_cmd_min

        self._positive_reward = positive_reward
        self._sym_kernel = sym_kernel
        self._step_len_min = step_len_min
        self._step_len_speed_exp = step_len_speed_exp
        self._step_len_v_ref = step_len_v_ref
        self._airtime_cutoff = airtime_cutoff
        self._w_place = w_place
        self._place_min = place_min
        self._w_footflat = w_footflat
        self._footflat_deadband = footflat_deadband
        self._w_jlim = w_jlim
        self._jlim_soft = jlim_soft
        self._jlim_joints = jlim_joints
        self._stand_prob = stand_prob
        self._gait_gate_v = gait_gate_v
        self._w_stand = w_stand
        self._cmd_dead_zone = cmd_dead_zone

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
            + (speed_range is not None)
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size
        }
        self._floor_id = self.model.geom("floor").id
        self._foot_ids = (self.model.geom("foot_geom").id, self.model.geom("foot_left_geom").id)
        self._body_weight = self.model.body_mass.sum() * -self.model.opt.gravity[2]

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    def set_style_scale(self, s):
        self._style_scale = float(s)

    def set_speed_range(self, lo, hi):
        self._speed_range = (float(lo), float(hi))     

    def symmetry_error(self):
        # Mean squared difference between the right leg now and the left leg half a cycle earlier (and vice versa).
        if len(self._recent_step_times) < 2:
            return None
        # half period = mean of the last two step times
        half = int(round(np.mean(self._recent_step_times) / self.dt))
        half = max(half, self._sym_min_half)
        if half < 1 or half >= len(self._joint_history):
            return None
        now, then = self._joint_history[-1], self._joint_history[-1 - half]
        return float(0.5 * (np.mean((now[0:3] - then[3:6]) ** 2)
                            + np.mean((now[3:6] - then[0:3]) ** 2)))

    def foot_contacts(self):
        # Whether each foot touches the floor, [right, left]
        con = self.data.contact
        pairs = np.asarray(con.geom).reshape(-1, 2)
        with_floor = (pairs == self._floor_id).any(axis=1)
        right = bool(np.any(with_floor & (pairs == self._foot_ids[0]).any(axis=1)))
        left = bool(np.any(with_floor & (pairs == self._foot_ids[1]).any(axis=1)))
        return [right, left]

    def foot_forces(self): 
        # for calculating the normal contact forces between the feet and the ground
        f6, out = np.zeros(6), [0.0, 0.0]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = tuple(c.geom) if hasattr(c, "geom") else (c.geom1, c.geom2)
            if self._floor_id not in g:
                continue
            for k in (0, 1):
                if self._foot_ids[k] in g:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    out[k] += f6[0]
        return out
    
    def foot_clearance(self):
        # Height of the lowest point of each foot above the floor, [right, left].
        out = []
        for gid in self._foot_ids:
            c = self.data.geom_xpos[gid]
            axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]       
            r, half = self.model.geom_size[gid][0], self.model.geom_size[gid][1]
            z_low = min(c[2] + half * axis[2], c[2] - half * axis[2]) - r
            out.append(float(z_low))
        return out

    def detect_touchdown(self):
        # A touchdown = the foot touches the floor after >= min_air_steps in the air.
        # Returns a dict with touchdown ("", "R", "L" or "both"); 
        # for an alternating step also step_time, step_length, swing_height, air_time and step_time_error.
        contacts = self.foot_contacts()
        clr = self.foot_clearance()
        foot_x = (self.data.body("foot").xpos[0], self.data.body("foot_left").xpos[0])
        t = self._step_count * self.dt
        result = dict(touchdown="", air_time=np.nan, step_time=np.nan, step_length=np.nan,
                      step_time_error=np.nan, swing_height=np.nan)

        landed, air, single, swing = [], {}, {}, {}
        for i in (0, 1):                               # 0 = right foot, 1 = left foot
            if not contacts[i]:                                           # in the air: track the highest clearance
                self._swing_h[i] = max(self._swing_h[i], clr[i])
            if (contacts[i]
                and not self._in_contact[i]
                and self._air_steps[i] >= self._min_air_steps):
                landed.append(i)
                air[i] = self._air_steps[i] * self.dt
                single[i] = self._single_steps[i] * self.dt
                swing[i] = self._swing_h[i]                              
            if contacts[i]:
                self._swing_h[i] = 0.0
            self._air_steps[i] = 0 if contacts[i] else self._air_steps[i] + 1
            self._contact_steps[i] = self._contact_steps[i] + 1 if contacts[i] else 0
            self._single_steps[i] = (0 if contacts[i]
                                     else self._single_steps[i] + int(contacts[1 - i]))
            self._in_contact[i] = contacts[i]

        if len(landed) == 2:                  #if both feet landed simultaneously
            result["touchdown"] = "both"
            self._last_touchdown = None
            self._last_step = None
            self._recent_step_times.clear()   # double-foot landing: SYM = 0 until two alternating steps
            self._td_steps["R"].clear(); self._td_steps["L"].clear()
            return result
        if not landed:
            return result

        i = landed[0]
        foot = "R" if i == 0 else "L"
        result["touchdown"] = foot
        self._td_steps[foot].append(self._step_count)
        result["air_time"] = air[i]
        result["single_air_time"] = single[i]
        result["swing_height"] = swing[i]
        last = self._last_touchdown
        result["same_foot"] = last is not None and last[0] == foot
        if last is not None and last[0] != foot:
            result["step_time"] = t - last[1]
            self._recent_step_times.append(result["step_time"])
            result["step_length"] = foot_x[i] - last[2]
            result["front_offset"] = foot_x[i] - self.data.body("thigh").xpos[0]   # landing foot relative to the hip
            prev = self._last_step
            result["prev_step_length"] = prev[1] if prev is not None else np.nan

            sgn = -1.0 if (self._target_speed is not None and self._target_speed < 0) else 1.0
            if prev is not None and sgn * result["step_length"] > 0.05 and sgn * prev[1] > 0.05:
                tau, tau_prev = result["step_time"], prev[0]
                result["step_time_error"] = (tau - tau_prev) / ((tau + tau_prev) / 2)
            self._last_step = (result["step_time"], result["step_length"])
        else:
            self._last_step = None
        self._last_touchdown = (foot, t, foot_x[i])
        return result

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._speed_range is not None:
            observation = np.concatenate((position, velocity, [self._target_speed])).ravel()
        else:
            observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self._step_count += 1
        touchdown_info = self.detect_touchdown()
        self._touchdown = touchdown_info
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            "target_speed": np.nan if self._target_speed is None else self._target_speed,
            **reward_info,
            **touchdown_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        # task: velocity tracking + healthy
        if self._target_speed is None:
            forward_reward = self._forward_reward_weight * x_velocity
        else:
            err = (x_velocity - self._target_speed) / self._track_sigma
            forward_reward = self._forward_reward_weight * np.exp(-err ** 2)
        # healthy reward: constant bonus while torso height and pitch are within the healthy range
        healthy_reward = self.healthy_reward

        # sym
        self._joint_history.append(self.data.qpos[3:9].copy())
        self._sym_error = self.symmetry_error()
        if self._sym_error is None:
            sym_reward, sym_norm, amp_ratio = 0.0, np.nan, np.nan
        else:
            q = np.array(self._joint_history)
            var = np.var(q, axis=0) ## per joint, over the window
            amp = var.mean()
            sym_norm = self._sym_error / (amp + self._sym_eps)
            e = sym_norm if self._sym_normalize else self._sym_error
            if self._sym_kernel == "inv":
                sym_reward = self._w_sym / (1.0 + self._sym_k * e)     
            else:
                sym_reward = self._w_sym * np.exp(-self._sym_k * e)     
            vr, vl = var[0:3].mean(), var[3:6].mean()
            amp_ratio = (min(vr, vl) + self._sym_eps) / (max(vr, vl) + self._sym_eps)
            if self._sym_amp_ratio:
                sym_reward *= amp_ratio

        # flight
        F_R, F_L = self.foot_forces()
        airborne = (F_R + F_L) < self._flight_force_ratio * self._body_weight
        flight_penalty = self._w_flight * float(airborne)

        # touchdown quantities (only on an alternating R/L touchdown)
        foot = self._touchdown.get("touchdown", "")
        tau = self._touchdown.get("step_time", np.nan)
        alternating = foot in ("R", "L") and np.isfinite(tau)

        # height
        height_reward = 0.0
        h = self._touchdown.get("swing_height", np.nan)
        if alternating and np.isfinite(h):
            height_reward = self._w_height * tau * min(h / self._height_target, 1.0)

        # air
        airtime_reward = 0.0
        cmd_ok = self._target_speed is None or abs(self._target_speed) > self._airtime_cmd_min
        c = self._in_contact
        if self._w_airtime > 0 and cmd_ok and c[0] != c[1]:
            t_mode = [(self._contact_steps[i] if c[i] else self._air_steps[i]) * self.dt for i in (0, 1)]
            if self._airtime_cutoff:      # Spot style: a mode that lasts longer than the threshold earns 0
                f = [t if t < self._airtime_threshold else 0.0 for t in t_mode]
                airtime_reward = self._w_airtime * min(f)
            else:                         # biped style: capped at the threshold
                airtime_reward = self._w_airtime * min(min(t_mode), self._airtime_threshold)

        
        # stride
        stride_penalty = 0.0
        if self._w_stride > 0 and foot in ("R", "L"):
            steps = self._td_steps[foot]
            if len(steps) >= 2:
                stride = (steps[-1] - steps[-2]) * self.dt          # same foot: touchdown -> next touchdown
                stride_penalty = self._w_stride * max(self._stride_min - stride, 0.0)

        # step
        step_penalty = 0.0
        ell = self._touchdown.get("step_length", np.nan)
        prev = self._touchdown.get("prev_step_length", np.nan)
        direction = 1.0 if (self._target_speed is None or self._target_speed >= 0) else -1.0
        if self._w_cross > 0 and self._step_len_min > 0 and self._touchdown.get("same_foot", False):
            step_penalty = self._w_cross                 
        if alternating and np.isfinite(ell):
            a = direction * ell
            if self._w_cross > 0:
                if self._step_len_min > 0:
                    L_min = self._step_len_min
                    if self._step_len_speed_exp != 0 and self._target_speed is not None:
                        L_min *= (abs(self._target_speed) / self._step_len_v_ref) ** self._step_len_speed_exp
                    if L_min > 1e-3:
                        step_penalty = self._w_cross * float(np.clip((L_min - a) / L_min, 0.0, 1.0))
                elif a <= 0.0:                             
                    step_penalty = self._w_cross

        # place: landing foot should land ahead of the hip
        place_penalty = 0.0
        off = self._touchdown.get("front_offset", np.nan)
        if self._w_place > 0 and alternating and np.isfinite(off):
            place_penalty = self._w_place * float(np.clip((self._place_min - direction * off) / self._place_min, 0.0, 1.0))


        # footflat: stance foot should be roughly flat on the ground
        footflat_penalty = 0.0
        if self._w_footflat > 0:
            for gid, on in zip(self._foot_ids, self.foot_contacts()):
                if on:
                    ax = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]          # foot capsule axis (ankle -> toe)
                    deg = np.degrees(np.arctan2(abs(ax[2]), abs(ax[0])))       # 0 = flat
                    footflat_penalty += self._w_footflat * max(deg - self._footflat_deadband, 0.0) / 45.0
        # jlim: joints should not sit at their hard limits (legged_gym / Isaac Lab dof_pos_limits)
        jlim_penalty = 0.0
        if self._w_jlim > 0:
            rng = self.model.jnt_range[3:9]                           # [6, 2] in rad
            mid, half = rng.mean(axis=1), 0.5 * (rng[:, 1] - rng[:, 0])
            lo, hi = mid - self._jlim_soft * half, mid + self._jlim_soft * half
            q = self.data.qpos[3:9]
            viol = np.clip(lo - q, 0.0, None) + np.clip(q - hi, 0.0, None)   # rad beyond the soft limit
            jlim_penalty = self._w_jlim * float(viol[list(self._jlim_joints)].sum())

        # tsym 
        tsym_reward = 0.0
        e_t = self._touchdown.get("step_time_error", np.nan)
        if self._w_tsym > 0 and not np.isnan(e_t):
            tsym_reward = self._w_tsym * tau * np.exp(-0.5 * (e_t / self._tsym_sigma) ** 2)

        standing_cmd = (self._target_speed is not None
                and abs(self._target_speed) < self._gait_gate_v)
        if standing_cmd:
            sym = height = tsym = airtime = 0.0          
            stride = place = 0.0      
            stand_penalty = self._w_stand * np.mean(np.abs(self.data.qpos[3:9] - self.init_qpos[3:9]))
        else:
            stand_penalty = 0.0

        # style with curriculum 
        s = self._style_scale
        penalties = flight_penalty + step_penalty + stand_penalty # + stride_penalty + place_penalty + footflat_penalty + jlim_penalty
        if self._curriculum_penalties_only:
            style = sym_reward + s * (height_reward + tsym_reward+ airtime_reward - penalties)
        else:
            style = s * (sym_reward + height_reward + tsym_reward+ airtime_reward - penalties)
    
        rewards = forward_reward + healthy_reward + style

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        reward = rewards - costs
        if self._positive_reward:
            reward = max(reward, 0.0)  

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "reward_symmetry": sym_reward,
            "sym_error": np.nan if self._sym_error is None else self._sym_error,
            "sym_error_norm": sym_norm,
            "sym_amp_ratio": amp_ratio,
            "airborne": airborne,
            "reward_flight": - flight_penalty,
            "reward_height": height_reward,
            "reward_stride": -stride_penalty,
            "reward_step": -step_penalty,
            "reward_tsym": tsym_reward,
            "reward_airtime": airtime_reward,
            "reward_place":  -place_penalty,
            "reward_footflat": -footflat_penalty,
            "reward_jlim": -jlim_penalty,
            "reward_stand": -stand_penalty,
        }

        return reward, reward_info

    def reset_model(self):
        self._joint_history.clear()
        self._recent_step_times.clear()
        self._td_steps["R"].clear(); self._td_steps["L"].clear()
        self._step_count = 0
        self._sym_error = None
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self._swing_h = [0.0, 0.0]
        self._single_steps = [0, 0]

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self._in_contact = self.foot_contacts()       
        self._air_steps = [0, 0]
        self._contact_steps = [0, 0]
        self._last_touchdown = None
        self._last_step = None
        self._touchdown = {}
        if self._speed_range is not None:
            self._target_speed = float(self.np_random.uniform(*self._speed_range))
            if abs(self._target_speed) < self._cmd_dead_zone:
                self._target_speed = 0.0                      # set small commands to zero
        if self._stand_prob > 0 and self.np_random.random() < self._stand_prob:
            self._target_speed = 0.0
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
def make_env(max_episode_steps=1000, **env_kwargs):
    return TimeLimit(MyWalkerEnv(**env_kwargs), max_episode_steps=max_episode_steps)
