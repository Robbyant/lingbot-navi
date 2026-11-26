import math, time, re
from pathlib import Path
import cv2
import magnum as mn
import numpy as np
from .init_explore import ExploreMap, pose_xy_heading
from .localization_tf import camera_in_odom
from .routes import relative_execution_path, simplify_follow_polyline

def _slug(value):
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_') or 'scene'

def jpeg(bgr):
    (ok, data) = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return data.tobytes()

class BaselinePolicy:

    def _capture_tf_history_sample(self):
        """Keep level RGB/odometry keyframes only during initial exploration."""
        if not getattr(self, '_init_explore_active', False):
            return
        from .tf_initialization import should_capture, HISTORY_LIMIT
        quality_sampling = self.config.get('dataset') == 'lingbot'
        from .tf_frame_sampling import depth_quality
        samples = self._tf_history_samples
        odom = camera_in_odom(*self._sensor_pose())
        if not should_capture(samples, self.step, odom):
            return
        try:
            nodes = [self.agent._sensors[name].node for name in (('rgb', 'depth') if quality_sampling else ('rgb',))]
            previous = [node.rotation for node in nodes]
            try:
                for node in nodes:
                    node.rotation = mn.Quaternion.identity_init()
                odom = camera_in_odom(*self._sensor_pose())
                observations = self.sim.get_sensor_observations()
                image = jpeg(cv2.cvtColor(observations['rgb'][:, :, :3], cv2.COLOR_RGB2BGR))
                quality = depth_quality(observations['depth']) if quality_sampling else None
            finally:
                for (node, rotation) in zip(nodes, previous):
                    node.rotation = rotation
            root = self.config['global_plan_work']
            path = Path(root) / 'tf_history' / f'step_{self.step:06d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
        except (OSError, RuntimeError, ValueError) as error:
            return
        samples.append((self.step, path, odom.copy()))
        if quality_sampling:
            self._tf_history_quality[self.step] = quality
        else:
            del samples[:-HISTORY_LIMIT]

    def _initialize_task_tf(self, scene, memories, *, stage='subtask'):
        """Initial exploration only: keep the existing diverse-station Huber fit."""
        if stage != 'initial_exploration':
            return False
        from .tf_initialization import initialize_tf
        task_key = (scene, str(self.episode['episode_id']), self.subtask_idx)
        previous = self._localization_tf
        self._capture_tf_history_sample()

        expected = previous.target_frame if previous.target_block == str(memories[0]) else None
        try:
            (candidate, details) = initialize_tf(self._tf_history_samples, memories, lambda paths, image: self.global_planner.localize_view(scene, paths, image), expected_frame=expected, qualities=self._tf_history_quality if self.config.get('dataset') == 'lingbot' else None,
                position_tolerance_m=(2.0 if self.config.get('lingbot_map1', False) else 1.75) if self.config.get('dataset') == 'lingbot' else 1.25,
                strict_errors=self.config.get('dataset') == 'lingbot')
        except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            (candidate, details) = (None, dict(ready=False, reason=str(error), method='huber'))
        if task_key != (scene, str(self.episode['episode_id']), self.subtask_idx) or self._localization_tf is not previous:
            self._record_behavior('tf_initialization_discarded', '标定期间任务已切换', calibration_stage=stage)
            return False
        ready = candidate is not None
        if ready:
            self._localization_tf = candidate
        self._record_behavior('tf_task_initialization', '初始探索 Huber 标定完成，后续持续更新 TF' if ready else '初始化失败，回退原 TF 转换及更新', ready=ready, reason=details.get('reason'), samples=details.get('attempted'))
        return ready

    def _capture_tf_exploration_sample(self):
        self._capture_tf_history_sample()

    def _sync_tf_target(self, anchor, localization):
        """Track the exact local frame, including recentering of a merged anchor."""
        reference = localization.get('reference_block_from_odom')
        if reference is None:
            return
        tf = self._localization_tf
        anchor = str(anchor)
        frame = np.linalg.inv(np.asarray(reference, dtype=float))
        if tf.target_block is None:
            tf.rebase(np.eye(4), target_block=anchor, target_frame=frame)
        elif tf.target_block != anchor:
            raise ValueError('TF target changed without the all-memory predicted bridge')
        elif not np.allclose(tf.target_frame, frame, atol=1e-08, rtol=0):
            bridge = np.linalg.inv(frame) @ tf.target_frame
            tf.rebase(bridge, target_block=anchor, target_frame=frame)
            self._record_behavior('tf_rebase', '同一记忆块局部坐标更新', target_block=anchor, new_from_old=bridge.tolist())

    def _task_planning_memories(self, scene, pool):
        """One all-pool prediction per task; grow its locked subset with new data."""
        from .task_memory_selection import anchor_bridge, rigid_frame, memory_versions, append_task_memories
        task_key = (scene, str(self.episode['episode_id']), self.subtask_idx)
        state = getattr(self, '_task_memory_selection', None)
        if state is not None and state['key'] == task_key:
            (selected, versions, added) = append_task_memories(state['selected'], state['versions'], pool)
            selected = [p for p in selected if p not in state.get('removed', set())]
            added = [p for p in added if p in selected]
            changed = {p for p in versions if state['versions'].get(p) != versions[p]}
            state['prunable'] = [p for p in state.get('prunable', []) if p not in changed]
            state.update(selected=selected, versions=versions)
            return selected
        tf = self._localization_tf
        previous = tf.target_block
        if previous is None:
            import re
            base = f"ep{_slug(str(self.episode['episode_id']))}_init_explore"
            initial = [str(p) for p in pool if re.fullmatch(re.escape(base)+r'(?:_part[0-9]+of[0-9]+)?', Path(p).parent.name)]
            initial.sort(key=lambda p: Path(p).parent.name)
            if not initial:
                raise ValueError('missing initial exploration memory for TF calibration')
            previous = initial[0]
            self._initialize_tf_from_exploration(scene, previous, memories=initial)
            tf = self._localization_tf
        if previous not in list(map(str, pool)):
            raise ValueError('previous TF target is missing from the total memory pool')
        full = [previous, *(str(p) for p in pool if str(p) != previous)]
        rgb = self.config['global_plan_work'] / f'subset_ep{_slug(task_key[1])}_sub{self.subtask_idx + 1}.jpg'
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(self.observation_jpegs()[0])
        raw = self.global_planner.select_task_memory(scene, full, rgb)
        if task_key != (scene, str(self.episode['episode_id']), self.subtask_idx):
            raise RuntimeError('task changed during memory selection')
        selection = raw['selection']
        selected = selection['selected']
        frames = raw['reference_block_frames']
        index = selection['selected_indices'][0]
        old_frame = tf.target_frame if tf.target_frame is not None else rigid_frame(frames[0])
        bridge = anchor_bridge(old_frame, frames[0], raw['block_predictions'][index], frames[index], raw['selected_anchor_frame'])
        tf.rebase(bridge, target_block=selected[0], target_frame=rigid_frame(raw['selected_anchor_frame']))
        self._task_memory_selection = dict(key=task_key, selected=selected, versions=memory_versions(pool), prunable=list(selection['prunable_oldest_first']), removed=set(), low_gain_streak=0)
        return selected

    def _rank_tf_observation_view(self):
        """Choose one depth-ranked view for a periodic replan, without pose fitting."""
        from .observation_view import select_observation_view
        (episode, subtask, auto) = (self.episode, self.subtask_idx, self._auto_follow)

        def turn(action):
            outcome = self._apply_motion(action, ground=False)
            return outcome != 'timeout' and (not self.done) and (self.episode is episode) and (self.subtask_idx == subtask) and (self._auto_follow is auto)
        return select_observation_view(lambda : self._current_rgb_depth()[1], turn, turn_deg=self.turn_deg)

    def _select_observation_view(self, scene, memories):
        """Use each current-view localization to update the moving TF estimate."""
        heading = list(map(float, self._agent_model_pose()[1]))
        result = dict(mode='current_view', start_heading_xy=heading, heading_xy=heading, attempts=[], selected_rank=None)
        auto = getattr(self, '_auto_follow', None)
        if auto is not None and auto.pop('rank_tf_view', False):
            ranked = self._rank_tf_observation_view()
            if ranked is None:
                return None
            result.update(ranked, mode='ranked_single_view', selected_bin=ranked['bins'][0], heading_xy=list(map(float, self._agent_model_pose()[1])))
        rgb = self.config['global_plan_work'] / f'current_view_{self._plan_token + 1}.jpg'
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(self.observation_jpegs()[0])
        odom = camera_in_odom(*self._sensor_pose())
        try:
            raw = self.global_planner.localize_view(scene, memories, rgb)
        except (OSError, RuntimeError, ValueError) as error:
            raw = dict(pose=None, answer=str(error))
        self._sync_tf_target(memories[0], raw)
        self._localization_tf.update(odom, raw['pose'])
        estimated = self._localization_tf.matrix()
        pose = self._localization_tf.predict(odom)
        result['attempts'].append(dict(rank=1, answer=raw['answer']))
        probe = None
        if pose is not None:
            result['selected_rank'] = 1
            xyz = ','.join((f'{v:+.2f}' for v in pose[:3]))
            quat = ','.join((f'{v:+.4f}' for v in pose[3:]))
            probe = dict(pose=pose.tolist(), answer=f'The image pose is <|3d_pose|> at ({xyz}), quaternion_xyzw=({quat}).', prompt='Current RGB camera pose from filtered odometry.', block_from_odom=estimated.tolist())
        return (result, probe)

    def _plan_has_motion(self, mode):
        try:
            path = relative_execution_path(mode.get('world_xyz'), mode.get('predicted_start_pose_world'), [0.0, 0.0], [1.0, 0.0])
            return bool(np.linalg.norm(path[-1]) >= self.forward_m * 0.5)
        except (ValueError, TypeError):
            return False

    def _short_exploration_route(self, mode):
        if mode.get('mode') != 'explore' or mode.get('ok') is False:
            return False
        try:
            path = relative_execution_path(mode.get('world_xyz'), mode.get('predicted_start_pose_world'), [0.0, 0.0], [1.0, 0.0])
            return bool(np.isfinite(path).all() and np.linalg.norm(path[-1]) < self.forward_m * 0.5)
        except (ValueError, TypeError, IndexError):
            return False

    def _plan_mode_summaries(self, ui):
        modes = ui.get('modes') or []
        return [{'mode': mode.get('mode'), 'label': mode.get('label') or mode.get('mode'), 'text': mode.get('text') or '', 'prompt': mode.get('model_prompt') or '', 'response': mode.get('generated_text') or '', 'has_bev': False, 'candidate_priority': mode.get('candidate_priority'), 'can_follow': self._plan_has_motion(mode), 'can_recover_short_explore': self._short_exploration_route(mode)} for mode in modes]

    def _repeat_subtask_pending(self):
        return getattr(self, '_episode_active', False) and (not self.done) and (self.subtask_idx > 0) and (self.current_task() == self.episode['tasks'][self.subtask_idx - 1])

    def _submit_at_budget_boundary(self):
        if not getattr(self, '_episode_active', False) or self.done or self.subtask_step != self.budget - 1:
            return False
        self._submit_auto_task(f'子任务仅剩 1 步，执行 STOP（预算 {self.budget} 步）')
        return True

    def episode_step(self):
        if not getattr(self, '_episode_active', False):
            return
        if self.done:
            self._episode_active = False
            return
        if self._repeat_subtask_pending():
            self._submit_auto_task('与上一子任务类型和内容完全相同，原地交卷')
            if self.done:
                self._episode_active = False
            return
        if self._submit_at_budget_boundary():
            return
        if self._init_explore_active:
            self.init_explore_step()
            self._episode_wait_since = time.monotonic()
        elif self._follow['active']:
            self.follow_step()
            self._episode_wait_since = time.monotonic()
        else:
            state = self._plan_ui.get('state')
            modes = self._plan_mode_summaries(self._plan_ui)
            available = [m for m in modes if m.get('can_follow') or m.get('can_recover_short_explore')]
            if state == 'done' and available:
                mode = self._preferred_auto_mode(available)
                if mode is None:
                    return
                (episode, subtask) = (self.episode, self.subtask_idx)
                self.start_follow(mode)
                same_task = self.episode is episode and self.subtask_idx == subtask
                if same_task and (not self._follow['active']) and (not self.done):
                    self._submit_auto_task('规划无法执行')
            elif state == 'error' or (state == 'done' and (not available)) or time.monotonic() - self._episode_wait_since > 300:
                self._submit_auto_task('规划失败或等待超过 300 秒')
        if self.done:
            self._episode_active = False

    def _searching_text_task(self):
        return getattr(self, '_text_task_search', False) and (not getattr(self, '_task_has_retrieval', False))

    def _defer_search_timeout(self):
        auto = getattr(self, '_auto_follow', None) or {}
        return self._searching_text_task() and auto.get('mode') == 'explore' and (not getattr(self, '_task_exploration_closed', False))

    def _preferred_auto_mode(self, modes):
        available = {m.get('mode') for m in modes if m.get('can_follow') or m.get('can_recover_short_explore')}
        if 'image' in available:
            return 'image'
        if getattr(self, '_pre_task_explore_pending', False) and (not getattr(self, '_task_exploration_closed', False)):
            if 'explore' in available:
                return 'explore'
            self._submit_auto_task('前置探索未返回可执行轨迹')
            return None

        def density_priority(candidate):
            name = candidate.get('mode') or ''
            counts = candidate.get('candidate_point_counts') or []
            index = ord(name[-1]) - ord('A') if name.startswith('dialogue-') and name[-1:].isupper() else -1
            count = counts[index] if 0 <= index < len(counts) else 0
            return (candidate['candidate_priority'] if candidate.get('candidate_priority') is not None else -count, name)
        for candidate in sorted(modes, key=density_priority):
            name = candidate.get('mode') or ''
            if name.startswith('dialogue-') and name in available and (not self._a_goal_blocked(name)):
                return name
        if getattr(self, 'config', {}).get('allow_direct_goal', True) and 'dialogue' in available and (not self._a_goal_blocked('dialogue')):
            return 'dialogue'
        if getattr(self, '_task_exploration_closed', False):
            self._submit_auto_task('最后一次探索记忆已检索，未返回可执行目标，不再继续探索')
            return None
        if self._searching_text_task() and self.subtask_step > 400:
            self._submit_auto_task('子任务超过 400 步，探索结束后仍未检索到可执行目标')
            return None
        if 'explore' in available:
            return 'explore'
        self._submit_auto_task('未返回可执行的检索目标或探索轨迹')
        return None

    def _a_goal_blocked(self, mode_name='dialogue-A'):
        blocked = getattr(self, '_blocked_a_goals', []) + getattr(self, '_visited_a_goals', [])
        if not blocked:
            return False
        mode = next((m for m in self._plan_ui.get('modes', []) if m.get('mode') == mode_name), None)
        if mode is None:
            return False
        try:
            (xy, heading) = self._agent_model_pose()
            goal = relative_execution_path(mode['world_xyz'], mode['predicted_start_pose_world'], xy, heading)[-1]
        except (ValueError, TypeError, KeyError):
            return False
        return any((np.linalg.norm(goal - np.asarray(point)) < 0.5 for point in blocked))

    def _finish_exploration_segment(self):
        self._task_explorations = getattr(self, '_task_explorations', 0) + 1
        self._pre_task_explore_pending = False
        if self.subtask_step >= self.budget:
            self._stop_follow_for_timeout()
            return
        final_retrieval = self._task_explorations >= 10 or (self._searching_text_task() and self.subtask_step > 400)
        if final_retrieval:
            self._task_exploration_closed = True
            self._auto_follow.pop('exploration_memory_retry', None)
            self._auto_follow['final_exploration_retrieval'] = True
            self._record_behavior('final_retrieval', '探索上限已到，写入最后观察后仅补一次检索，不再探索')
        phase = f'explore_sub{self.subtask_idx + 1}_{self._task_explorations}'
        self._record_behavior('memory', f'探索 {self._task_explorations} 次完成，写入独立记忆 {phase}')
        auto = self._auto_follow
        auto.pop('explore_target', None)
        auto['memory_phase'] = phase
        auto['wait_since'] = time.monotonic()
        self._follow['action'] = ''
        self._follow['local_route'] = []
        self._follow['local_status'] = '探索完成，等待独立记忆写入后重规划'
        try:
            auto['coverage_before'] = self._exploration_pool_cells()
            self._finish_memory_segment(phase=phase)
            self._restart_sim_geom()
        except Exception as error:
            self._submit_auto_task(f'探索记忆提交失败：{error}')

    def _finish_short_exploration(self):
        """An unexecuted short route is a zero-gain attempt, not a task stop."""
        if self.subtask_step >= self.budget:
            self._stop_follow_for_timeout()
            return
        auto = self._auto_follow
        self._task_explorations = getattr(self, '_task_explorations', 0) + 1
        self._pre_task_explore_pending = False
        auto.pop('exploration_memory_retry', None)
        auto.pop('explore_target', None)
        if self._task_explorations >= 10 or (self._searching_text_task() and self.subtask_step > 400):
            self._task_exploration_closed = True
            auto['final_exploration_retrieval'] = True
            self._record_behavior('final_retrieval', '过短探索达到上限，仅补一次最终检索')
        self._record_behavior('explore_end', '探索轨迹不足半步，按零新增面积恢复')
        auto['coverage_before'] = self._exploration_pool_cells()
        self._apply_exploration_gain(auto)
        self._replan_follow('探索轨迹过短，按零新增面积继续恢复')

    def _exploration_pool_cells(self):
        from .coverage import memory_coverage
        from .task_memory_selection import memory_versions
        pool = self._current_scene_memories()
        versions = memory_versions(pool)
        cache = getattr(self, '_exploration_coverage_cache', {})
        for (path, version) in versions.items():
            if path not in cache or cache[path][0] != version:
                cache[path] = (version, memory_coverage([path])[path])
        self._exploration_coverage_cache = {p: cache[p] for p in versions}
        return set().union(*(cache[p][1] for p in versions))

    def _apply_exploration_gain(self, auto):
        from .task_memory_selection import record_exploration_gain
        before = auto.pop('coverage_before')
        after = self._exploration_pool_cells()
        gain = len(after - before) * 0.25 ** 2
        scene = self.cases[self.case_pos]['scene_name']
        self._task_planning_memories(scene, self._current_scene_memories())
        state = self._task_memory_selection
        removed = record_exploration_gain(state, gain)
        self._record_behavior('exploration_coverage_gain', '探索写入后的总池净新增覆盖', added_area_m2=gain, before_area_m2=len(before) * 0.25 ** 2, after_area_m2=len(after) * 0.25 ** 2, low_gain_streak=state['low_gain_streak'], removed_memory=removed, target_block=state['selected'][0])
        if removed:
            self._record_behavior('explore_memory_reduce', '连续两轮新增面积不足 1 平方米，删除最旧的基准前记忆', removed_memory=removed, remaining_memories=len(state['selected']))
            if not auto.get('final_exploration_retrieval'):
                auto['exploration_memory_retry'] = {'memories': list(state['selected'])}
        elif state['low_gain_streak'] >= 2:
            self._task_exploration_closed = True
            auto.pop('exploration_memory_retry', None)
            if not auto.get('final_exploration_retrieval'):
                self._record_behavior('final_retrieval', '连续两轮探索新增面积不足 1 平方米且无可删记忆，仅补一次最终检索，不再探索')
            auto['final_exploration_retrieval'] = True

    def _begin_endpoint_scan(self):
        """Observe a full turn at the predicted endpoint before submitting."""
        auto = self._auto_follow
        if auto.get('endpoint_scan'):
            return
        turns = max(1, int(math.ceil(360.0 / self.turn_deg)))
        auto['endpoint_scan'] = {'remaining': turns, 'total': turns}
        self._follow['action'] = 'turn_left'
        self._follow['local_status'] = f'终点环视 0/{turns}，完成后交卷'
        self._record_behavior('endpoint_scan_start', f'到达预测终点，原地左转 {turns} 步后交卷')

    def _step_endpoint_scan(self):
        auto = self._auto_follow
        scan = auto['endpoint_scan']
        result = self._apply_motion('turn_left', ground=False)
        if result == 'timeout' or self.done or self._auto_follow is not auto:
            return
        self._follow['steps'] += 1
        scan['remaining'] -= 1
        completed = scan['total'] - scan['remaining']
        self._follow['local_status'] = f"终点环视 {completed}/{scan['total']}，完成后交卷"
        if scan['remaining'] <= 0:
            self._record_behavior('endpoint_scan_done', '预测终点全周环视完成')
            self._submit_auto_task('已到预测终点，环视完成后交卷')

    def _blank_follow(self):
        return {'active': False, 'mode': '', 'label': '', 'action': '', 'steps': 0, 'blocked': 0, 'reason': '', 'polyline': None}

    def _plan_world_path(self, mode):
        modes = list(self._plan_ui.get('modes') or [])
        selected = next((item for item in modes if item.get('mode') == mode), None)
        if not selected or not selected.get('world_xyz'):
            return ('', None, None)
        return (str(selected.get('label') or mode), selected.get('world_xyz'), selected.get('predicted_start_pose_world'))

    def start_follow(self, mode, *, continuation=False):
        """Anchor model-relative motion at the current robot pose, then track it."""
        mode = str(mode).strip()
        if self._follow['active']:
            return
        if self._init_explore_active:
            return
        if self.done:
            return
        if mode == 'explore' and getattr(self, '_task_exploration_closed', False):
            auto = getattr(self, '_auto_follow', None)
            if auto is not None:
                auto.pop('exploration_memory_retry', None)
            self._submit_auto_task('探索已关闭，拒绝再次执行探索')
            return
        label, world, predicted_pose = self._plan_world_path(mode)
        actual_xy, actual_heading = self._agent_model_pose()
        try:
            execution_path = relative_execution_path(world, predicted_pose, actual_xy, actual_heading)
            polyline = simplify_follow_polyline(execution_path)
        except (ValueError, TypeError) as error:
            self._follow = self._blank_follow()
            self._follow['mode'] = mode
            self._follow['reason'] = str(error)
            return
        if len(polyline) == 0:
            self._follow = self._blank_follow()
            self._follow['mode'] = mode
            self._follow['reason'] = '没有轨迹'
            return
        if mode == 'dialogue' or mode.startswith('dialogue-'):
            picked = next((m for m in self._plan_ui.get('modes', []) if m.get('mode') == mode), {})
            self._previous_goal_xy = picked.get('selected_world_xy')
        if (mode == 'dialogue' or mode.startswith('dialogue-')) and getattr(self, '_text_task_search', False):
            if not getattr(self, '_task_has_retrieval', False):
                auto = getattr(self, '_auto_follow', None)
                if auto is not None:
                    auto['replans'] = 0
            self._task_has_retrieval = True
        if not continuation:
            self._auto_follow = {'mode': mode, 'replans': 0, 'stalls': 0, 'distance': float(np.linalg.norm(polyline[-1] - np.asarray(actual_xy))), 'waiting': False, 'travelled': 0.0, 'stuck_retry': False}
            self._auto_follow['watch_distance'] = self._auto_follow['distance']
            self._auto_follow['watch_step'] = self.subtask_step
        auto = getattr(self, '_auto_follow', None)
        if mode == 'explore' and auto is not None:
            if auto.get('explore_target') is None:
                auto['explore_motion_m'] = 0.0
                auto['explore_target'] = [float(polyline[-1, 0]), float(polyline[-1, 1]), float(self.agent.get_state().position[1])]
                auto['watch_distance'] = float(np.linalg.norm(polyline[-1] - np.asarray(actual_xy)))
                auto['watch_step'] = self.subtask_step
            else:
                polyline[-1] = np.asarray(auto['explore_target'][:2])
        elif auto is not None:
            auto.pop('explore_target', None)
        if auto is not None:
            auto['segment_forward_attempts'] = 0
            auto['segment_motion_m'] = 0.0
            if auto.get('failure_replans') and (not auto.get('failure_retry_executing')):
                auto['failure_retry_executing'] = True
                auto['failure_retry_start_step'] = self.subtask_step
            auto['watch_distance'] = float(np.linalg.norm(polyline[-1] - np.asarray(actual_xy)))
        self._follow = {'active': True, 'mode': mode, 'label': label or mode, 'action': '', 'steps': 0, 'blocked': 0, 'reason': '', 'polyline': polyline, 'model_execution_path': execution_path}
        if auto is not None:
            auto.pop('watch_waypoint', None)
        self._record_behavior('select', f'执行 {label or mode}', goal=polyline[-1].tolist(), continuation=continuation, mode=mode)
        if mode == 'explore' and np.linalg.norm(execution_path[-1] - np.asarray(actual_xy)) < self.forward_m * 0.5:
            self._finish_short_exploration()
            return
        self._begin_plan_alignment(actual_heading)

    def follow_step(self):
        pending_auto = getattr(self, '_auto_follow', None) or {}
        'Take one atomic step along the active plan. Does not ground this frame.'
        if not self._follow['active']:
            return
        if self.done:
            self._end_follow('episode 已结束', ground=False)
            return
        if self.subtask_step >= self.budget and (not self._defer_search_timeout()):
            self._stop_follow_for_timeout()
            return
        auto = getattr(self, '_auto_follow', None)
        if auto and auto.get('endpoint_scan'):
            self._step_endpoint_scan()
            return
        if auto and auto.get('memory_phase'):
            self._wait_exploration_memory(auto)
            return
        if auto and (not auto.get('waiting')):
            if (auto['mode'] == 'dialogue' or auto['mode'].startswith('dialogue-')) and auto.get('failure_replans', 0) >= 1 and (self.subtask_step - auto.get('failure_retry_start_step', self.subtask_step) >= 200):
                self._submit_auto_task('失败重规划后 200 步仍未到预测终点')
                return
            xy, _ = self._agent_model_pose()
            target = np.asarray(self._follow['polyline'])[-1]
            if auto['mode'] == 'explore' and auto.get('explore_target') is not None:
                target = np.asarray(auto['explore_target'][:2])
            progress_distance = float(np.linalg.norm(target - np.asarray(xy)))
            nav = self._follow.get('endpoint_follower')
            if nav is not None:
                progress_distance = nav.diagnostics.get('remaining_geodesic_m', progress_distance)
                waypoint = nav.diagnostics.get('waypoint_index')
                if auto.get('watch_waypoint') != waypoint:
                    auto['watch_waypoint'] = waypoint
                    auto['watch_distance'] = None
                    auto['watch_step'] = self.subtask_step
            previous = auto.get('watch_distance')
            if progress_distance is not None and math.isfinite(progress_distance):
                if previous is None or previous - progress_distance >= 0.15:
                    auto['watch_distance'] = progress_distance
                    auto['watch_step'] = self.subtask_step
                    auto['stuck_retry'] = False
                if self.subtask_step - auto.get('watch_step', self.subtask_step) >= 200:
                    self._retry_stuck('连续 200 步未接近终点')
                    return
        if auto and auto.get('waiting'):
            state = self._plan_ui.get('state')
            token = self._plan_token
            if token != auto.get('token'):
                self._end_follow('自动重规划已被替换')
            elif state == 'done':
                modes = self._plan_mode_summaries(self._plan_ui)
                final_retrieval = auto.pop('final_exploration_retrieval', False)
                if final_retrieval or getattr(self, '_task_exploration_closed', False):
                    auto.pop('exploration_memory_retry', None)
                    self._pre_task_explore_pending = False
                    mode = self._preferred_auto_mode(modes)
                elif auto.get('exploration_memory_retry') is not None:
                    mode = next((m['mode'] for m in modes if m.get('mode') == 'explore' and (m.get('can_follow') or m.get('can_recover_short_explore'))), None)
                    if mode is None:
                        self._retry_exploration_memory('删减记忆后未返回可执行探索轨迹')
                        return
                elif auto.get('failure_replans', 0) >= 1:
                    compatible = [auto['mode']]
                    if auto['mode'] == 'dialogue' or auto['mode'].startswith('dialogue-'):
                        compatible = [m['mode'] for m in sorted(modes, key=lambda m: (m.get('candidate_priority') is None, m.get('candidate_priority'))) if m.get('mode') == 'dialogue' or str(m.get('mode', '')).startswith('dialogue-')]
                    mode = next((name for name in compatible if any((m.get('mode') == name and m.get('can_follow') for m in modes))), None)
                    if mode is None:
                        self._submit_auto_task('本次重规划未返回同类可执行轨迹，结束当前执行')
                        return
                else:
                    mode = self._preferred_auto_mode(modes)
                if mode is None:
                    return
                if mode != auto['mode']:
                    auto['watch_distance'] = None
                    auto['watch_step'] = self.subtask_step
                    auto['stuck_retry'] = False
                auto['mode'] = mode
                auto['waiting'] = False
                self._follow['active'] = False
                self.start_follow(mode, continuation=True)
                if not self._follow['active']:
                    self._auto_follow = None
            elif state == 'error' and auto.get('exploration_memory_retry') is not None:
                error = str(self._plan_ui.get('text') or '规划未返回轨迹')
                self._retry_exploration_memory('删减记忆后的规划失败：' + error)
            elif state in {'error', 'waiting'} or time.monotonic() - auto.get('wait_since', time.monotonic()) > 300:
                self._submit_auto_task('自动重规划失败或超时')
            return
        if not self._follow['action']:
            self._refresh_follow_action()
            if not self._follow['active'] or not self._follow.get('action'):
                return
        follow = self._follow
        action = follow['action']
        before_xy, before_heading = self._agent_model_pose()
        self._last_motion_collided = False
        outcome = self._apply_motion(action, ground=False)
        follow['steps'] += 1
        if outcome == 'timeout':
            return
        after_xy, _ = self._agent_model_pose()
        if outcome == 'floor_blocked':
            follow['action'] = ''
            follow['blocked'] = 0
            self._finish_follow_segment('固定楼层限制拒绝上下楼')
            return
        if outcome == 'terrain_blocked':
            follow['action'] = ''
            follow['blocked'] = 0
            self._refresh_follow_action()
            return
        hit = action == 'move_forward' and (outcome == 'blocked' or self._last_motion_collided)
        if auto:
            if action == 'move_forward':
                auto['segment_forward_attempts'] = auto.get('segment_forward_attempts', 0) + 1
            auto['segment_motion_m'] = auto.get('segment_motion_m', 0.0) + float(np.linalg.norm(np.asarray(after_xy) - before_xy))
        if auto:
            moved = float(np.linalg.norm(np.asarray(after_xy) - before_xy))
            auto['travelled'] = auto.get('travelled', 0.0) + moved
            auto['retry_motion'] = auto.get('retry_motion', 0.0) + moved
            if auto['mode'] == 'explore':
                auto['explore_motion_m'] = auto.get('explore_motion_m', 0.0) + moved
            if auto.get('stuck_retry') and auto.get('retry_kind', 'local_motion') == 'local_motion' and (auto['retry_motion'] >= 0.45):
                auto['stuck_retry'] = False
            if auto['mode'] != 'explore' and auto['travelled'] >= 2.0:
                self._replan_follow('累计移动 2 m', periodic=True)
                return
        if hit:
            self._follow['blocked'] += 1
        else:
            self._follow['blocked'] = 0
        if self._follow['blocked'] >= 3:
            self._finish_follow_segment('前进被挡住')
            return
        self._refresh_follow_action()

    def _new_endpoint_follower(self):
        raise NotImplementedError("Dataset session must provide a navigation executor")

    def _refresh_endpoint_follow_action(self):
        follow = self._follow
        planner = follow.get('endpoint_follower')
        if planner is None:
            planner = self._new_endpoint_follower()
            follow['endpoint_follower'] = planner
        endpoint = follow['polyline'][-1:]
        if follow.get('mode') == 'explore':
            if 'endpoint_exploration_goal' not in follow:
                goal, detail = planner.exploration_endpoint(follow['polyline'], step_m=self.forward_m)
                if goal is None:
                    follow['action'] = ''
                    follow['local_route'] = []
                    follow['local_status'] = '探索路线沿途无可前进的可达终点'
                    self._retry_exploration_memory(follow['local_status'])
                    return
                follow['endpoint_exploration_goal'] = np.asarray(goal).reshape(1, 2)
                auto = getattr(self, '_auto_follow', None)
                if auto is not None:
                    auto['explore_target'] = [float(goal[0]), float(goal[1]), float(self.agent.get_state().position[1])]
                    auto['watch_distance'] = detail['geodesic_m']
                    auto['watch_step'] = self.subtask_step
            endpoint = follow['endpoint_exploration_goal']
        elif 'endpoint_backtrack_goal' in follow:
            endpoint = follow['endpoint_backtrack_goal']
        action, reason = planner.next_action(endpoint)
        if action is None and follow.get('mode') != 'explore' and ('endpoint_backtrack_goal' not in follow) and (planner.diagnostics.get('error') in {'occ_no_progress', 'no_reachable_projection_within_limits', 'no_path_on_episode_floor', 'waypoint_no_longer_reachable'}):
            goal, detail = planner.navigation_backtrack(follow['model_execution_path'])
            if goal is not None:
                endpoint = follow['endpoint_backtrack_goal'] = goal.reshape(1, 2)
                action, reason = planner.next_action(endpoint)
        follow['action'] = action
        follow['local_status'] = planner.status
        follow['local_route'] = planner.route
        detail = {**planner.diagnostics, 'action': action, 'reason': reason, 'episode': str(self.episode['episode_id']), 'subtask': self.subtask_idx + 1, 'step': self.subtask_step}
        executor = 'habitat'
        if action is None:
            if 'endpoint_backtrack_goal' in follow and planner.diagnostics.get('completion') in {'model_endpoint', 'projected_endpoint'}:
                auto = getattr(self, '_auto_follow', None)
                if auto is None:
                    self._end_follow('已到临时可达点', ground=False)
                else:
                    self._replan_follow('已到临时可达点，重新验证新规划终点')
            elif planner.diagnostics.get('completion') == 'model_endpoint':
                self._finish_follow_segment('已到轨迹终点')
            elif planner.diagnostics.get('completion') == 'projected_endpoint':
                self._finish_follow_segment('预测终点不可达，已到可达替代终点')
            elif follow.get('mode') == 'explore' and (not follow.get('steps', 0)):
                if (getattr(self, '_auto_follow', None) or {}).get('exploration_memory_retry') is not None:
                    self._retry_exploration_memory(reason or '探索路径尚未执行即失败')
                else:
                    self._submit_auto_task(reason or '探索路径尚未执行即失败')
            else:
                self._finish_follow_segment(reason or '路径执行失败')
        elif (getattr(self, '_auto_follow', None) or {}).get('exploration_memory_retry') is not None:
            recovery = self._auto_follow.pop('exploration_memory_retry')
            self._record_behavior('explore_memory_recovered', '删减记忆后找到可执行探索路径', remaining_memories=len(recovery['memories']))

    def _finish_follow_segment(self, reason):
        """Replan after route completion or a recoverable local failure."""
        if reason != '已到轨迹终点':
            self._retry_stuck(reason)
            return
        auto = getattr(self, '_auto_follow', None)
        if auto is None or self.done:
            self._end_follow(reason)
            return
        if self.subtask_step >= self.budget and (not self._defer_search_timeout()):
            self._stop_follow_for_timeout()
            return
        if auto['mode'] == 'explore':
            self._finish_exploration_segment()
            return
        self._begin_endpoint_scan()

    def _retry_stuck(self, reason):
        auto = getattr(self, '_auto_follow', None)
        if auto is None or self.done:
            self._end_follow(reason)
            return
        if auto['mode'] == 'explore':
            self._record_behavior('explore_end', '本次探索停止：' + reason)
            self._finish_exploration_segment()
            return
        if not (auto['mode'] == 'dialogue' or auto['mode'].startswith('dialogue-')):
            self._submit_auto_task('当前模式不做失败重试：' + reason)
            return
        if auto.get('failure_replans', 0) >= 1 or auto.get('stuck_retry'):
            self._submit_auto_task('重规划后仍无法接近终点：' + reason)
            return
        auto['failure_replans'] = auto.get('failure_replans', 0) + 1
        auto['failure_retry_start_step'] = self.subtask_step
        auto['stuck_retry'] = True
        auto['retry_kind'] = 'task_progress' if '200 步未接近终点' in reason else 'local_motion'
        auto['retry_motion'] = 0.0
        auto['watch_step'] = self.subtask_step
        self._begin_retry_memory(reason)

    def _begin_retry_memory(self, reason):
        auto = self._auto_follow
        phase = f"retry_sub{self.subtask_idx + 1}_{auto['failure_replans']}"
        auto['memory_phase'] = phase
        auto['memory_retry_reason'] = reason
        auto['wait_since'] = time.monotonic()
        self._follow['action'] = ''
        self._follow['local_route'] = []
        self._follow['local_status'] = '失败重试前，等待 spatial memory 写入'
        self._record_behavior('memory', f'失败重试前写入独立记忆 {phase}')
        try:
            self._finish_memory_segment(phase=phase)
            self._restart_sim_geom()
        except Exception as error:
            self._submit_auto_task(f'失败重试前记忆提交失败：{error}')

    def _replan_follow(self, reason, *, periodic=False):
        auto = getattr(self, '_auto_follow', None)
        if auto is None or self.done:
            return
        if not periodic and auto['replans'] >= 12 and (not self._searching_text_task()) and (not auto.get('final_exploration_retrieval')) and (auto.get('exploration_memory_retry') is None):
            self._submit_auto_task('达到重规划上限')
            return
        (xy, _) = self._agent_model_pose()
        path = self._follow.get('polyline')
        distance = None if path is None else float(np.linalg.norm(np.asarray(path)[-1] - np.asarray(xy)))
        auto['rank_tf_view'] = periodic
        auto['last_replan_reason'] = reason
        auto['travelled'] = 0.0
        auto['wait_since'] = time.monotonic()
        if periodic:
            auto['periodic_replans'] = auto.get('periodic_replans', 0) + 1
        elif auto.get('exploration_memory_retry') is None:
            auto['replans'] += 1
        auto['waiting'] = True
        self._follow['action'] = ''
        self._follow['local_route'] = []
        recovery = auto.get('exploration_memory_retry')
        planning = '最优视角单帧更新 TF 后重新规划' if periodic else '删减记忆，当前视角定位后重新规划' if recovery is not None else '当前视角定位后重新规划'
        self._follow['local_status'] = planning
        self._record_behavior('replan', reason + '；' + planning, round=auto['replans'], periodic=periodic, periodic_replans=auto.get('periodic_replans', 0))
        try:
            self._request_global_plan(force=True)
        except Exception as error:
            self._submit_auto_task(f'自动重规划失败：{error}')
            return
        if getattr(self, '_auto_follow', None) is auto:
            auto['token'] = self._plan_token

    def _stop_follow_for_timeout(self):
        self._follow['active'] = False
        self._follow['action'] = ''
        self._follow['reason'] = '步数用尽'
        self._follow['polyline'] = None
        self._finish_subtask('timeout')

    def _end_follow(self, reason, *, ground=True):
        self._auto_follow = None
        label = self._follow['label'] or self._follow['mode']
        steps = int(self._follow['steps'])
        mode = self._follow['mode']
        self._follow['active'] = False
        self._follow['action'] = ''
        self._follow['reason'] = reason
        self._follow['polyline'] = None
        self._follow['local_route'] = []
        self._follow['local_status'] = ''

    def _arm_init_explore(self):
        self._init_explore_starting = True
        try:
            return self._arm_init_explore_impl()
        finally:
            self._init_explore_starting = False

    def _arm_init_explore_impl(self):
        """Start one cold-start roam when this scene has no spatial memory."""
        self._init_explore_active = False
        if self.done:
            return
        scene = self.cases[self.case_pos]['scene_name']
        episode_key = (scene, str(self.episode['episode_id']))
        if episode_key in self._init_explored_scenes:
            return
        if self.config.get('spatial_memory', True) and self._current_scene_memories():
            return
        camera = self.camera['rgb']
        self._explore_map = ExploreMap(visibility_height=(1.4, 1.6), hfov_deg=float(camera['hfov']), forward_m=self.forward_m, turn_deg=self.turn_deg)
        self._init_endpoint_follower = None
        self._init_endpoint_candidate_planner = None
        self._init_endpoint_goal = None
        self._init_endpoint_reference = None
        self._init_endpoint_watch = None
        self._init_endpoint_hits = 0
        self._init_explore_steps = 0
        self._init_explore_limit = self.config['init_explore_max_steps']
        if not self._has_independent_initial_exploration_budget():
            remaining = self.budget - self.subtask_step - self.config['init_explore_navigation_reserve']
            self._init_explore_limit = min(self._init_explore_limit, max(0, remaining))
        self._init_explored_scenes.add(episode_key)
        self._init_explore_blocked = 0
        self._explore_samples = []
        self._init_explore_look_steps = 1
        if self._apply_motion('look_down', ground=False, phase='init_explore_setup') == 'timeout':
            return
        self._integrate_explore_view()
        self._update_frontier_state()
        self._init_explore_active = True

    def _integrate_explore_view(self):
        if self._explore_map is None:
            return
        (_rgb, depth) = self._current_rgb_depth()
        (sensor_xyz, sensor_quat) = self._sensor_pose()
        feet_y = float(self.agent.get_state().position[1])
        camera = self.camera['rgb']
        self._explore_map.integrate_depth(depth, sensor_xyz, sensor_quat, feet_y, width=int(depth.shape[1]), height=int(depth.shape[0]), hfov_deg=float(camera['hfov']))
        xy = pose_xy_heading(sensor_xyz, sensor_quat)[0]
        if not getattr(self, '_last_motion_collided', False):
            self._explore_map.confirm_footprint(xy)

    def _update_frontier_state(self):
        # status() stamps/caches frontier state; preserve this navigation side effect.
        if self._explore_map is not None:
            self._explore_map.status()


    def _init_frontier_route(self, goal):
        planner = getattr(self, '_init_endpoint_candidate_planner', None)
        if planner is None or planner.sim is not self.sim:
            planner = self._new_endpoint_follower()
            self._init_endpoint_candidate_planner = planner
        return planner.project_route(goal)

    def _choose_init_endpoint_action(self, xy, heading):
        grid = self._explore_map
        grid._external_route_planner = self._init_frontier_route
        for _ in range(6):
            intent = grid.choose_intent(xy, heading)
            if intent['kind'] != 'navigate':
                self._init_endpoint_follower = None
                self._init_endpoint_goal = None
                return intent['action']
            goal = np.asarray(intent['goal'], dtype=float)
            if self._init_endpoint_goal is None or not np.allclose(goal, self._init_endpoint_goal, atol=1e-06):
                self._init_endpoint_goal = goal.copy()
                self._init_endpoint_reference = np.array([xy, goal], dtype=float)
                self._init_endpoint_follower = self._new_endpoint_follower()
                self._init_endpoint_watch = (self.step, float('inf'))
                self._init_endpoint_hits = 0
            planner = self._init_endpoint_follower
            (action, reason) = planner.next_action(self._init_endpoint_reference)
            self._explore_map.plan_xy = [np.asarray(point) for point in planner.route]
            if action is None:
                if planner.diagnostics.get('completion') in ('model_endpoint', 'projected_endpoint'):
                    grid.navigation_arrived(xy)
                else:
                    grid.navigation_failed(reason or 'Endpoint follower failed')
                self._init_endpoint_goal = None
                self._init_endpoint_follower = None
                continue
            remaining = planner.diagnostics.get('remaining_geodesic_m', float('inf'))
            (watch_step, best) = self._init_endpoint_watch
            if best - remaining >= 0.1:
                self._init_endpoint_watch = (self.step, remaining)
            elif self.step - watch_step >= 40:
                grid.navigation_failed('40 个动作未接近当前 frontier 观察点')
                self._init_endpoint_goal = None
                self._init_endpoint_follower = None
                continue
            return action
        return grid._scan_for_frontier(xy, heading, pending=bool(grid.frontier_clusters(include_pending=True)))

    def _update_init_habitat_motion(self, action, outcome, xy, heading):
        if action != 'move_forward':
            return
        if outcome == 'floor_blocked':
            self._explore_map.navigation_failed('固定楼层限制拒绝上下楼')
            self._init_endpoint_follower = None
            self._init_endpoint_goal = None
            self._init_endpoint_hits = 0
            self._init_explore_blocked = 0
            return
        after = np.asarray(self._agent_model_pose()[0])
        if np.linalg.norm(after - np.asarray(xy)) > 1e-06:
            self._explore_map.mark_traversed(xy, after)
        if outcome == 'blocked':
            self._init_explore_blocked += 1
            self._init_endpoint_hits += 1
            self._explore_map.mark_blocked_ahead(after, heading)
            if self._init_endpoint_hits >= 3:
                self._explore_map.navigation_failed('当前观察点执行发生三次碰撞')
                self._init_endpoint_follower = None
                self._init_endpoint_goal = None
                self._init_endpoint_hits = 0
        else:
            self._init_explore_blocked = 0
            self._init_endpoint_hits = 0

    def init_explore_step(self):
        """Integrate the current depth, take one coverage step, then replan."""
        if not self._init_explore_active or self.done or self._explore_map is None:
            return
        if self._init_explore_steps >= self._init_explore_limit:
            reason = '达到探索步数上限，保留后续导航预算'
            self._end_init_explore(reason)
            return
        self._capture_tf_exploration_sample()
        self._integrate_explore_view()
        (sensor_xyz, sensor_quat) = self._sensor_pose()
        (xy, heading) = pose_xy_heading(sensor_xyz, sensor_quat)
        action = self._choose_init_endpoint_action(xy, heading)
        if any((event['event'] == 'select_viewpoint' for event in self._explore_map.decisions)):
            self._explore_samples = []
        if action is None:
            self._update_frontier_state()
            self._end_init_explore(self._explore_map.stop_reason or '没有还能靠近的 frontier')
            return
        outcome = self._apply_motion(action, ground=False, phase='init_explore')
        self._init_explore_steps += 1
        if outcome == 'timeout' or not self._init_explore_active:
            return
        self._update_init_habitat_motion(action, outcome, xy, heading)
        self._update_frontier_state()
