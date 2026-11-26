"""Single-question decoding with center/pose feedback and trajectory dispatch.

Accepts one prompt, one scene memory and optional user-supplied goal XYZ.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .constants import CENTER_TOKEN, POSE_TOKEN, SCENE_TOKEN, TRAJECTORY_TOKEN, VIEWPOINT_TOKEN
from .model.joint import Branch, insert_after, insert_state_after


@dataclass
class JointGenerationResult:
    generated_text: str
    sampled_token_ids: list[int]
    forced_token_ids: list[int]
    predicted_centers: list
    predicted_poses: list
    trajectory: list | None
    finish_reason: str
    predicted_viewpoints: list | None = None
    point_predictions: list | None = None
    point_local_predictions: list | None = None
    block_predictions: list | None = None
    coordinate_frame: str = "first_block_meters"
    block_coordinate_frame: str = "first_block_meters"


@torch.no_grad()
def generate_joint(
    adapter,
    *,
    prompt_input_ids,
    prompt_inputs_embeds,
    prompt_attention_mask,
    prompt_position_ids,
    query_scene_indices,
    memory_xyz,
    memory_features,
    block_cu_seqlens,
    scene_block_cu_seqlens,
    navigation=None,
    goal_xyz=None,
    image_goal=None,
    candidate_centers=None,
    viewpoint_grounding=None,
    initial_pose=None,
    max_sampled_tokens=256,
    max_centers=10,
    use_cache=True,
    return_block_diagnostics=False,
    return_point_predictions=False,
    encoded_scenes=None,
    scene_memory_cache=None,
):
    if adapter.training:
        raise ValueError("call model.eval() before autoregressive evaluation")
    if max_sampled_tokens <= 0 or max_centers <= 0:
        raise ValueError("generation limits must be positive")
    batch_size = prompt_input_ids.shape[0]
    if batch_size != 1:
        raise ValueError("Only one query is supported")
    navigation = list(navigation) if navigation is not None else [False] * batch_size
    if len(navigation) != batch_size:
        raise ValueError("navigation flags must align with prompts")
    viewpoint_grounding = list(viewpoint_grounding) if viewpoint_grounding is not None else [True] * batch_size
    if len(viewpoint_grounding) != batch_size:
        raise ValueError("viewpoint grounding flags must align with prompts")
    from .prompts import canonical_goal

    goals = list(goal_xyz) if goal_xyz is not None else [None] * batch_size
    if len(goals) != batch_size:
        raise ValueError("goal_xyz must align with prompts")
    goals = [canonical_goal(goal) if goal is not None else None for goal in goals]
    image_goals = list(image_goal) if image_goal is not None else [False] * batch_size
    if len(image_goals) != batch_size:
        raise ValueError("image_goal flags must align with prompts")
    if any((image and goal is not None for image, goal in zip(image_goals, goals))):
        raise ValueError("image-goal navigation cannot also specify goal_xyz")
    navigation = [
        nav or image or goal is not None
        for nav, image, goal in zip(navigation, image_goals, goals)
    ]
    supplied = (
        list(candidate_centers)
        if candidate_centers is not None
        else [None] * batch_size
    )
    if len(supplied) != batch_size:
        raise ValueError("candidate_centers must align with prompts")
    from .prompts import candidate_prefixes

    prefixes = [
        candidate_prefixes(values) if values is not None else [] for values in supplied
    ]
    if any((nav and fields for nav, fields in zip(navigation, prefixes))):
        raise ValueError("inline candidates are incompatible with navigation")
    candidate_index = [0] * batch_size
    description_parts = [""] * batch_size
    spatial = {
        "memory_xyz": memory_xyz,
        "memory_features": memory_features,
        "block_cu_seqlens": block_cu_seqlens,
        "scene_block_cu_seqlens": scene_block_cu_seqlens,
    }
    if encoded_scenes is not None and scene_memory_cache is not None:
        raise ValueError("provide either encoded_scenes or scene_memory_cache")
    if encoded_scenes is not None:
        spatial["_scene_cache"] = encoded_scenes
    elif scene_memory_cache is not None:
        spatial["_scene_cache"] = scene_memory_cache.get(
            adapter, spatial, prompt_inputs_embeds
        )
    branches, prompt_lengths = ([], [])
    for i in range(batch_size):
        valid = prompt_attention_mask[i].bool()
        ids = prompt_input_ids[i, valid]
        if not ids.numel():
            raise ValueError("generation prompt is empty")
        branches.append(
            Branch(
                ids,
                prompt_inputs_embeds[i, valid],
                prompt_position_ids[:, i, valid],
                int(query_scene_indices[i]),
            )
        )
        prompt_lengths.append(ids.numel())
    sampled, forced = ([[] for _ in branches], [[] for _ in branches])
    centers, poses = ([[] for _ in branches], [[] for _ in branches])
    viewpoints = [[] for _ in branches]
    initial = list(initial_pose) if initial_pose is not None else [None] * batch_size
    if len(initial) != batch_size:
        raise ValueError("initial_pose must align with prompts")
    from .feedback import canonical_pose

    for i, value in enumerate(initial):
        if value is not None:
            if goals[i] is None:
                raise ValueError(
                    "initial_pose is only for coordinate-navigation continuation"
                )
            poses[i].append(canonical_pose(prompt_inputs_embeds.new_tensor(value, dtype=torch.float32)))
    paths, finished = ([None for _ in branches], [None for _ in branches])
    token_ids = adapter.token_ids
    center_id, pose_id, path_id = (
        token_ids[t] for t in (CENTER_TOKEN, POSE_TOKEN, TRAJECTORY_TOKEN)
    )
    viewpoint_id = token_ids.get(VIEWPOINT_TOKEN)
    eos = adapter.tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, (list, tuple)) else [eos])
    cache = {} if use_cache else None

    def append(branch, ids):
        ids = branch.ids.new_tensor(ids)
        if branch.ids.numel() + ids.numel() > adapter.max_context_tokens:
            return False
        insert_after(branch, branch.ids.numel() - 1, ids, adapter.embed_token_ids(ids))
        return True

    def append_fixed(i, text):
        ids = adapter.tokenizer.encode(text, add_special_tokens=False)
        if not append(branches[i], ids):
            finished[i] = "max_context_tokens"
            return False
        forced[i].extend(ids)
        return True

    for i, fields in enumerate(prefixes):
        if fields:
            append_fixed(i, fields[0])
    for _ in range(max_sampled_tokens):
        active = [i for i in range(batch_size) if finished[i] is None]
        if not active:
            break
        hidden, _ = adapter._run_branches(
            [branches[i] for i in active], spatial, scene_cache=cache
        )
        for i in active:
            if navigation[i] and (not branches[i].cache["image_end"]):
                raise ValueError("navigation requires a current RGB image")
        dispatch = []
        for j, i in enumerate(active):
            branch = branches[i]
            logits = adapter.logits(hidden[j][-1]).float()
            logits[token_ids[SCENE_TOKEN]] = torch.finfo(logits.dtype).min
            if len(centers[i]) >= max_centers:
                logits[center_id] = torch.finfo(logits.dtype).min
            if prefixes[i]:
                for forbidden in (center_id, pose_id, path_id, viewpoint_id):
                    if forbidden is None:
                        continue
                    logits[forbidden] = torch.finfo(logits.dtype).min
            token = int(logits.argmax())
            if prefixes[i]:
                sampled[i].append(token)
                if token in stops:
                    if not append(branch, [token]):
                        finished[i] = "max_context_tokens"
                    elif (
                        candidate_index[i] != len(prefixes[i]) - 1
                        or not description_parts[i].strip()
                    ):
                        finished[i] = "invalid_protocol"
                    else:
                        finished[i] = "eos"
                    continue
                chunk = adapter.tokenizer.decode([token], skip_special_tokens=False)
                if "\n" in chunk:
                    before = chunk.split("\n", 1)[0]
                    description_parts[i] += before
                    last = candidate_index[i] == len(prefixes[i]) - 1
                    accepted = before + ("" if last else "\n")
                    ids = adapter.tokenizer.encode(accepted, add_special_tokens=False)
                    if ids and (not append(branch, ids)):
                        finished[i] = "max_context_tokens"
                        continue
                    if not description_parts[i].strip():
                        finished[i] = "invalid_protocol"
                        continue
                    if last:
                        end_id = next(iter(stops))
                        if append(branch, [end_id]):
                            forced[i].append(end_id)
                            finished[i] = "eos"
                        else:
                            finished[i] = "max_context_tokens"
                    else:
                        candidate_index[i] += 1
                        description_parts[i] = ""
                        append_fixed(i, prefixes[i][candidate_index[i]])
                else:
                    description_parts[i] += chunk
                    if not append(branch, [token]):
                        finished[i] = "max_context_tokens"
                continue
            if not append(branch, [token]):
                finished[i] = "max_context_tokens"
                continue
            sampled[i].append(token)
            if token in stops:
                finished[i] = "eos"
                continue
            if token in (center_id, pose_id, path_id, viewpoint_id):
                if viewpoint_id is not None and not navigation[i] and viewpoint_grounding[i]:
                    if ((token == viewpoint_id and len(centers[i]) != len(viewpoints[i]) + 1)
                            or (token == center_id and len(centers[i]) != len(viewpoints[i]))):
                        finished[i] = "invalid_protocol"
                        continue
                if navigation[i]:
                    expected = (
                        pose_id
                        if not poses[i]
                        else center_id
                        if not centers[i]
                        else pose_id
                        if len(poses[i]) == 1
                        else path_id
                        if paths[i] is None
                        else None
                    )
                    if image_goals[i]:
                        expected = (
                            pose_id
                            if len(poses[i]) < 2
                            else path_id
                            if paths[i] is None
                            else None
                        )
                    if goals[i] is not None:
                        expected = (
                            pose_id
                            if not poses[i]
                            else path_id
                            if paths[i] is None
                            else None
                        )
                    if token != expected:
                        finished[i] = "invalid_protocol"
                        continue
                if token == path_id:
                    expected_poses = 1 if goals[i] is not None else 2
                    if len(poses[i]) != expected_poses or paths[i] is not None:
                        finished[i] = "invalid_protocol"
                        continue
                    if branch.ids.numel() + 78 > adapter.max_context_tokens:
                        finished[i] = "max_context_tokens"
                        continue
                    queries = (
                        adapter.prompt_embedding.trajectory_queries
                        + adapter.prompt_embedding.trajectory_progress_encoding.to(
                            adapter.prompt_embedding.trajectory_queries
                        )
                    ).to(branch.embeds)
                    insert_after(
                        branch,
                        branch.ids.numel() - 1,
                        branch.ids.new_full((78,), -1),
                        queries,
                    )
                dispatch.append((i, token))
        if dispatch:
            output, prefix = adapter._run_branches(
                [branches[i] for i, _ in dispatch], spatial, scene_cache=cache
            )
            for j, (i, token) in enumerate(dispatch):
                branch = branches[i]
                if token == path_id:
                    goal_poses = poses[i]
                    if goals[i] is not None:
                        goal_poses = [
                            poses[i][0],
                            output[j].new_tensor(
                                [*goals[i], 0.0, 0.0, 0.0, 1.0], dtype=torch.float32
                            ),
                        ]
                    path = adapter.decode_trajectory(
                        output[j][-78:], torch.stack(goal_poses)
                    )
                    paths[i] = path
                    # Trajectory queries terminate navigation; there is no trailing LM target.
                    finished[i] = "trajectory_complete"
                elif token == viewpoint_id:
                    value = adapter._decode_value(output[j][-1], "viewpoint")
                    norm = value[3:].norm()
                    if not torch.isfinite(value).all() or norm < 1e-8:
                        finished[i] = "invalid_protocol"
                        continue
                    viewpoints[i].append(torch.cat((value[:3], value[3:] / norm)))
                else:
                    kind = "center" if token == center_id else "pose"
                    value, head_hidden = adapter._decode_value_features(
                        output[j][-1], kind
                    )
                    (centers[i] if kind == "center" else poses[i]).append(value)
                    if kind == "center" and adapter.feedback_mode != "text":
                        if branch.ids.numel() + 1 > adapter.max_context_tokens:
                            finished[i] = "max_context_tokens"
                            continue
                        insert_state_after(
                            branch,
                            branch.ids.numel() - 1,
                            adapter.feedback_embedding(kind, value, head_hidden),
                        )
                    else:
                        text = adapter.format_text_feedback(kind, value)
                        ids = adapter.tokenizer.encode(text, add_special_tokens=False)
                        if not append(branch, ids):
                            finished[i] = "max_context_tokens"
                        else:
                            forced[i].extend(ids)
    point_predictions = {}
    point_local_predictions = {}
    if return_point_predictions:
        if adapter.active_point_head is None:
            raise ValueError("point predictions require POINT_HEAD_ENABLED=1")
        point_spatial = dict(spatial)
        point_spatial["_scene_cache"] = (
            encoded_scenes
            if encoded_scenes is not None
            else cache["scenes"]
            if cache is not None
            else adapter._encode_scene_memories(spatial, prompt_inputs_embeds)
        )
        if adapter.model_config.point_coordinate_frame == "block_local":
            from .model.point_head import transform_memory_points

            point_local_predictions = adapter._predict_memory_points(
                point_spatial, raw=True
            )
            point_predictions = transform_memory_points(
                point_local_predictions,
                {
                    scene: entry[1]
                    for scene, entry in point_spatial["_scene_cache"].items()
                },
                spatial["block_cu_seqlens"],
                spatial["scene_block_cu_seqlens"],
            )
        else:
            point_predictions = adapter._predict_memory_points(point_spatial)
    results = []
    for i, branch in enumerate(branches):
        reason = finished[i] or "max_sampled_tokens"
        if (
            reason == "eos"
            and navigation[i]
            and (
                len(poses[i]) != (1 if goals[i] is not None else 2)
                or len(centers[i])
                != (0 if goals[i] is not None or image_goals[i] else 1)
                or paths[i] is None
            )
        ):
            reason = "invalid_protocol"
        if (reason == "eos" and viewpoint_id is not None and not navigation[i]
                and viewpoint_grounding[i] and len(centers[i]) != len(viewpoints[i])):
            reason = "invalid_protocol"
        context_ids = [int(v) for v in branch.ids[prompt_lengths[i] :] if v >= 0]
        results.append(
            JointGenerationResult(
                generated_text=adapter.tokenizer.decode(
                    context_ids, skip_special_tokens=False
                ),
                sampled_token_ids=sampled[i],
                forced_token_ids=forced[i],
                predicted_centers=[v.cpu().tolist() for v in centers[i]],
                predicted_viewpoints=[v.cpu().tolist() for v in viewpoints[i]],
                predicted_poses=[v.cpu().tolist() for v in poses[i]],
                trajectory=paths[i].cpu().tolist() if paths[i] is not None else None,
                finish_reason=reason,
                point_predictions=point_predictions[branch.scene].cpu().tolist()
                if return_point_predictions
                else None,
                point_local_predictions=point_local_predictions[branch.scene]
                .cpu()
                .tolist()
                if point_local_predictions
                else None,
                coordinate_frame="first_block_meters",
                block_predictions=[
                    dict(
                        translation=p[0].detach().float().cpu().tolist(),
                        rotation_matrix=p[1].detach().float().cpu().tolist(),
                        scale=float(p[2].detach().float().exp().cpu())
                        if adapter.ego.task_config.learn_block_scale
                        else 1.0,
                    )
                    for p in adapter._block_outputs.get(id(branch), [])
                ]
                if return_block_diagnostics
                else None,
            )
        )
    if hasattr(adapter, "_block_outputs"):
        for branch in branches:
            adapter._block_outputs.pop(id(branch), None)
    return results
