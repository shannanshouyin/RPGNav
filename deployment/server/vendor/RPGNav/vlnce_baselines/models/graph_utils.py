from collections import defaultdict
import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import torch
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff

MAX_DIST = 30
MAX_STEP = 10
ST_DIR_SPECIAL = 0
ST_DIR_FORWARD = 1
ST_DIR_BACKWARD = 2
ST_DIR_UNEXPLORED = 3
# NOISE = 0.5

ST_TIME_BUCKET_NUM = 7

def bucketize_time_delta(delta):
    delta_arr = np.asarray(delta)
    buckets = np.zeros_like(delta_arr, dtype=np.int64)
    buckets[(delta_arr >= -5) & (delta_arr <= -2)] = 1
    buckets[delta_arr == -1] = 2
    buckets[delta_arr == 0] = 3
    buckets[delta_arr == 1] = 4
    buckets[(delta_arr >= 2) & (delta_arr <= 5)] = 5
    buckets[delta_arr >= 6] = 6
    return buckets

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    # heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
    # if b[1] < a[1]:
    #     heading = np.pi - heading
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def heading_from_quaternion(quat: np.array):
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

def estimate_cand_pos(pos, ori, ang, dis):
    cand_num = len(ang)
    cand_pos = np.zeros([cand_num, 3])

    ang = np.array(ang)
    dis = np.array(dis)
    ang = (heading_from_quaternion(ori) + ang) % (2 * np.pi)
    cand_pos[:, 0] = pos[0] - dis * np.sin(ang)    # x
    cand_pos[:, 1] = pos[1]                        # y
    cand_pos[:, 2] = pos[2] - dis * np.cos(ang)    # z
    return cand_pos


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y and x !=k and y != k:
                    t_dis = self._dis[x][y] + self._dis[y][k]
                    if t_dis < self._dis[x][k]:
                        self._dis[x][k] = t_dis
                        self._dis[k][x] = t_dis
                        self._point[x][k] = y
                        self._point[k][x] = y

        for x in self._dis:
            for y in self._dis:
                if x != y:
                    t_dis = self._dis[x][k] + self._dis[k][y]
                    if t_dis < self._dis[x][y]:
                        self._dis[x][y] = t_dis
                        self._dis[y][x] = t_dis
                        self._point[x][y] = k
                        self._point[y][x] = k

        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug):

        self.graph_nx = nx.Graph()

        self.node_pos = {}          # viewpoint to position (x, y, z)
        self.node_embeds = {}       # viewpoint to pano feature
        self.node_pano_embeds = {}  # viewpoint to 12-view pano features
        self.node_oris = {}         # viewpoint to simulator orientation
        self.node_stepId = {}
        self.node_parent = {}

        self.ghost_cnt = 0          # id to create ghost 
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_embeds = {}      # viewpoint to single_view feature
        self.ghost_stepId = {}
        self.ghost_fronts = {}      # viewpoint to front_vp id
        self.ghost_real_pos = {}    # for training
        self.has_real_pos = has_real_pos
        self.merge_ghost = merge_ghost
        self.ghost_aug = ghost_aug  # 0 ~ 1, noise level
        self.loc_noise = loc_noise

        self.shortest_path = None
        self.shortest_dist = None
        self.failed_ghost_fronts = defaultdict(set)
        self.ghosts_with_failed_front = set()
        
        self.node_stop_scores = {}  # viewpoint to stop_score
        # Point cloud samples per ghost viewpoint (list of (N,3) tensors), optional branch.
        self.ghost_pc_points = {}

    def _localize(self, qpos, kpos_dict, ignore_height=False, loc_noise=None):
        """
        Args:
            loc_noise: If provided, use this value; otherwise use self.loc_noise
        """
        if loc_noise is None:
            loc_noise = self.loc_noise
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():
            if ignore_height:
                dis = ((qpos[[0,2]] - kpos[[0,2]])**2).sum()**0.5
            else:
                dis = ((qpos - kpos)**2).sum()**0.5
            if dis < min_dis:
                min_dis = dis
                min_vp = kvp
        min_vp = None if min_dis > loc_noise else min_vp
        return min_vp
    
    def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        # assume no repeated node
        # since action is restricted to ghosts
        cur_vp = str(len(self.node_pos))
        cand_vp = [f'{cur_vp}_{str(i)}' for i in range(len(cand_ang))]
        cand_pos = [p for p in estimate_cand_pos(cur_pos, cur_ori, cand_ang, cand_dis)]
        return cur_vp, cand_vp, cand_pos

    def delete_ghost(self, vp, remove_aug=False):
        self.ghost_pos.pop(vp)
        self.ghost_mean_pos.pop(vp)
        self.ghost_embeds.pop(vp)
        self.ghost_stepId.pop(vp, None)
        self.ghost_fronts.pop(vp)
        self.ghost_pc_points.pop(vp, None)
        self.failed_ghost_fronts.pop(vp, None)
        self.ghosts_with_failed_front.discard(vp)
        if remove_aug and hasattr(self, "ghost_aug_pos"):
            self.ghost_aug_pos.pop(vp, None)
        if self.has_real_pos:
            self.ghost_real_pos.pop(vp)

    def update_graph(self, prev_vp, step_id,
                           cur_vp, cur_pos, cur_embeds,
                           cand_vp, cand_pos, cand_embeds, 
                           cand_real_pos, cand_pc_points=None, loc_noise=None,
                           cur_pano_embeds=None, cur_ori=None):
        """
        Args:
            loc_noise: Dynamic loc_noise value, if provided use this value; otherwise use self.loc_noise
        """
        # 1. connect prev_vp
        self.graph_nx.add_node(cur_vp)
        if prev_vp is not None:
            prev_pos = self.node_pos[prev_vp]
            dis = calc_position_distance(prev_pos, cur_pos)
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis)

        # 2. update node & ghost info
        self.node_pos[cur_vp] = cur_pos
        self.node_embeds[cur_vp] = cur_embeds
        if cur_pano_embeds is not None:
            self.node_pano_embeds[cur_vp] = cur_pano_embeds
        if cur_ori is not None:
            self.node_oris[cur_vp] = cur_ori
        self.node_stepId.setdefault(cur_vp, step_id)
        self.node_parent.setdefault(cur_vp, prev_vp)
        # If dynamic loc_noise is provided, update self.loc_noise (for subsequent _localize calls)
        if loc_noise is not None:
            self.loc_noise = loc_noise
        for i, (cvp, cpos, cembeds) in enumerate(zip(cand_vp, cand_pos, cand_embeds)):
            cand_pc = None if cand_pc_points is None else cand_pc_points[i]
            localized_nvp = self._localize(cpos, self.node_pos, loc_noise=loc_noise)
            # cand overlap with node, connect cur_vp with localized_nvp
            if localized_nvp is not None :
                dis = calc_position_distance(cur_pos, self.node_pos[localized_nvp])
                self.graph_nx.add_edge(cur_vp, localized_nvp, weight=dis)
            # cand not overlap with node, create/update ghost
            else:
                if self.merge_ghost:
                    localized_gvp = self._localize(cpos, self.ghost_mean_pos, loc_noise=loc_noise)
                    # create ghost
                    if localized_gvp is None:
                        gvp = f'g{str(self.ghost_cnt)}'
                        self.ghost_cnt += 1
                        self.ghost_pos[gvp] = [cpos]
                        self.ghost_mean_pos[gvp] = cpos
                        self.ghost_embeds[gvp] = [cembeds, 1]
                        self.ghost_stepId[gvp] = step_id
                        self.ghost_fronts[gvp] = [cur_vp]
                        if cand_pc is not None:
                            self.ghost_pc_points[gvp] = [cand_pc]
                        if self.has_real_pos:
                            self.ghost_real_pos[gvp] = [cand_real_pos[i]]
                    # update ghost
                    else:
                        gvp = localized_gvp
                        self.ghost_pos[gvp].append(cpos)
                        self.ghost_mean_pos[gvp] = np.mean(self.ghost_pos[gvp], axis=0)
                        self.ghost_embeds[gvp][0] = self.ghost_embeds[gvp][0] + cembeds
                        self.ghost_embeds[gvp][1] += 1
                        self.ghost_fronts[gvp].append(cur_vp)
                        if cand_pc is not None:
                            self.ghost_pc_points.setdefault(gvp, []).append(cand_pc)
                        if self.has_real_pos:
                            self.ghost_real_pos[gvp].append(cand_real_pos[i])
                else:
                    gvp = f'g{str(self.ghost_cnt)}'
                    self.ghost_cnt += 1
                    self.ghost_pos[gvp] = [cpos]
                    self.ghost_mean_pos[gvp] = cpos
                    self.ghost_embeds[gvp] = [cembeds, 1]
                    self.ghost_stepId[gvp] = step_id
                    self.ghost_fronts[gvp] = [cur_vp]
                    if cand_pc is not None:
                        self.ghost_pc_points[gvp] = [cand_pc]
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp] = [cand_real_pos[i]]
        
        self.ghost_aug_pos = deepcopy(self.ghost_mean_pos)
        if self.ghost_aug != 0:
            for gvp, gpos in self.ghost_aug_pos.items():
                gpos_noise = np.random.normal(loc=(0,0,0), scale=(self.ghost_aug,0,self.ghost_aug), size=(3,))
                gpos_noise[gpos_noise < -self.ghost_aug] = -self.ghost_aug
                gpos_noise[gpos_noise >  self.ghost_aug] =  self.ghost_aug
                self.ghost_aug_pos[gvp] = gpos + gpos_noise

        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))

    def get_valid_ghost_fronts(self, ghost_vp):
        failed_fronts = self.failed_ghost_fronts.get(ghost_vp, set())
        valid_fronts = []
        seen_fronts = set()
        for front_vp in self.ghost_fronts.get(ghost_vp, []):
            if front_vp in failed_fronts or front_vp in seen_fronts:
                continue
            valid_fronts.append(front_vp)
            seen_fronts.add(front_vp)
        return valid_fronts

    def ghost_has_valid_front(self, ghost_vp):
        return len(self.get_valid_ghost_fronts(ghost_vp)) > 0

    def mark_ghost_front_failed(self, ghost_vp, front_vp):
        self.failed_ghost_fronts[ghost_vp].add(front_vp)
        self.ghosts_with_failed_front.add(ghost_vp)

    def ghost_had_failed_front(self, ghost_vp):
        return ghost_vp in self.ghosts_with_failed_front or bool(self.failed_ghost_fronts.get(ghost_vp, set()))

    def front_to_ghost_dist(self, ghost_vp, valid_only=False):
        # assume the nearest front
        min_dis = 10000
        min_front = None
        front_vps = self.get_valid_ghost_fronts(ghost_vp) if valid_only else self.ghost_fronts[ghost_vp]
        for front_vp in front_vps:
            dis = calc_position_distance(
                self.node_pos[front_vp], self.ghost_aug_pos[ghost_vp]
            )
            if dis < min_dis:
                min_dis = dis
                min_front = front_vp
        return min_dis, min_front

    def get_node_embeds(self, vp):
        if not vp.startswith('g'):
            return self.node_embeds[vp]
        else:
            return self.ghost_embeds[vp][0] / self.ghost_embeds[vp][1]

    def get_node_time(self, vp):
        if vp is None:
            return 0
        if vp.startswith('g'):
            return self.ghost_stepId.get(vp, 0)
        return self.node_stepId.get(vp, 0)

    def _nearest_ghost_front(self, ghost_vp):
        return self.front_to_ghost_dist(ghost_vp)[1]

    def _first_hop(self, src_vp, dst_vp):
        if src_vp is None or dst_vp is None or src_vp == dst_vp:
            return None
        if src_vp.startswith('g'):
            return self._nearest_ghost_front(src_vp)
        if dst_vp.startswith('g'):
            front_vp = self._nearest_ghost_front(dst_vp)
            if front_vp is None:
                return None
            if src_vp == front_vp:
                return dst_vp
            path = self.shortest_path.get(src_vp, {}).get(front_vp, None)
            return path[1] if path is not None and len(path) > 1 else None
        path = self.shortest_path.get(src_vp, {}).get(dst_vp, None)
        return path[1] if path is not None and len(path) > 1 else None

    def _direction_type(self, src_vp, dst_vp):
        hop = self._first_hop(src_vp, dst_vp)
        if hop is None:
            return ST_DIR_SPECIAL
        if hop.startswith('g'):
            return ST_DIR_UNEXPLORED
        if self.node_parent.get(hop, None) == src_vp:
            return ST_DIR_FORWARD
        if self.node_parent.get(src_vp, None) == hop:
            return ST_DIR_BACKWARD
        if self.get_node_time(hop) > self.get_node_time(src_vp):
            return ST_DIR_FORWARD
        return ST_DIR_BACKWARD

    def _vp_position(self, vp):
        if vp is None:
            return None
        if vp.startswith('g'):
            return self.ghost_aug_pos[vp]
        return self.node_pos[vp]

    def _view_feature_towards(self, src_vp, dst_vp, feature_dim, device, dtype):
        if src_vp is None or dst_vp is None or src_vp == dst_vp:
            return torch.zeros(feature_dim, device=device, dtype=dtype)
        if src_vp.startswith('g'):
            return self.get_node_embeds(src_vp).to(device=device, dtype=dtype)
        pano = self.node_pano_embeds.get(src_vp, None)
        src_ori = self.node_oris.get(src_vp, None)
        src_pos = self.node_pos.get(src_vp, None)
        dst_pos = self._vp_position(dst_vp)
        if pano is None or src_ori is None or src_pos is None or dst_pos is None:
            return self.get_node_embeds(src_vp).to(device=device, dtype=dtype)
        rel_heading, _, _ = calculate_vp_rel_pos_fts(
            src_pos, dst_pos, heading_from_quaternion(src_ori), 0, to_clock=False
        )
        rel_heading = rel_heading % (2 * np.pi)
        centers = (1 - np.arange(12, dtype=np.float32) / 12.0) * 2 * np.pi
        deltas = np.abs((centers - rel_heading + np.pi) % (2 * np.pi) - np.pi)
        view_idx = int(np.argmin(deltas))
        return pano[view_idx].to(device=device, dtype=dtype)

    def get_st_gasa_inputs(self, gmap_vp_ids, feature_dim, device, dtype, use_time_bucket=False):
        n = len(gmap_vp_ids)
        time_deltas = np.zeros((n, n), dtype=np.int64)
        dir_types = np.zeros((n, n), dtype=np.int64)
        dir_fts = torch.zeros(n, n, feature_dim, device=device, dtype=dtype)
        times = [self.get_node_time(vp) for vp in gmap_vp_ids]
        for i, src_vp in enumerate(gmap_vp_ids):
            for j, dst_vp in enumerate(gmap_vp_ids):
                time_deltas[i, j] = times[j] - times[i]
                dir_types[i, j] = self._direction_type(src_vp, dst_vp)
                hop = self._first_hop(src_vp, dst_vp)
                visual_vp = hop if hop is not None else dst_vp
                dir_fts[i, j] = self._view_feature_towards(
                    src_vp, visual_vp, feature_dim, device, dtype
                )
        if use_time_bucket:
            time_deltas = bucketize_time_delta(time_deltas)
        return time_deltas, dir_types, dir_fts

    def get_pos_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vp_ids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            # for ghost
            elif vp.startswith('g'):
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.ghost_aug_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                front_dis, front_vp = self.front_to_ghost_dist(vp)
                shortest_dist = self.shortest_dist[cur_vp][front_vp] + front_dis
                shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
            # for node
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
