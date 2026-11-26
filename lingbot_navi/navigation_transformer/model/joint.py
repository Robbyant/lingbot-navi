from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
from ..constants import SPATIAL_SPECIAL_TOKENS, STATE_TOKEN_ID, VIEWPOINT_TOKEN, TRAJECTORY_FLOW_PROTOCOL
from ..feedback import FeedbackPolicy, format_feedback, normalize_feedback_mode
from .configuration import BlockConfig
from .ego_blocks.model import LingBotNaviEgoBlockModel
from .ego_blocks.inputs import offsets
from .ego_blocks.pi3_pose import matrix_to_quaternion
from .prompt_metadata import image_bounds


@dataclass
class Branch:
    ids: Tensor
    embeds: Tensor
    positions: Tensor
    scene: int = 0
    cache: dict = field(default_factory=dict)
    cpu_ids: tuple | None = None


def insert_after(
    branch: Branch, position: int, ids: Tensor, embeds: Tensor, *, cpu_ids=None
):
    """Insert forced context or internal queries and keep every index aligned."""
    count = ids.numel()
    p = position + 1
    positions = (
        branch.positions[:, position : position + 1]
        + torch.arange(1, count + 1, device=branch.positions.device)[None]
    )
    branch.ids = torch.cat((branch.ids[:p], ids, branch.ids[p:]))
    branch.embeds = torch.cat((branch.embeds[:p], embeds, branch.embeds[p:]))
    branch.positions = torch.cat(
        (branch.positions[:, :p], positions, branch.positions[:, p:] + count), dim=1
    )
    if branch.cpu_ids is not None:
        inserted = (
            tuple(cpu_ids)
            if cpu_ids is not None
            else tuple(ids.detach().cpu().tolist())
        )
        branch.cpu_ids = branch.cpu_ids[:p] + inserted + branch.cpu_ids[p:]


def insert_state_after(branch: Branch, position: int, embedding: Tensor):
    """Force one internal state with its exact vector, without a base embedding."""
    insert_after(
        branch,
        position,
        branch.ids.new_full((1,), STATE_TOKEN_ID),
        embedding.to(branch.embeds).unsqueeze(0),
        cpu_ids=(STATE_TOKEN_ID,),
    )


class LingBotNaviJointAdapter(nn.Module):
    def _apply(self, fn, recurse=True):

        def preserve(value):
            converted = fn(value)
            if (
                value.is_floating_point()
                and value.dtype == torch.float32
                and (converted.dtype != torch.float32)
            ):
                return (
                    converted.float()
                    if value.is_meta
                    else value.to(device=converted.device)
                )
            return converted

        return super()._apply(preserve, recurse=recurse)

    @property
    def prompt_embedding(self):
        return self.ego.prompt_embedding

    @property
    def text_model(self):
        return self.ego.text_model

    @property
    def trajectory_head(self):
        return self.ego.trajectory_head

    @property
    def root_model(self):
        return object.__getattribute__(self, "_root_model")

    @property
    def active_point_head(self):
        return (
            self.global_point_head
            if self.global_point_head is not None
            else self.point_head
        )

    def embed_token_ids(self, ids):
        return self._overlay_embeddings(
            ids, self.root_model.get_input_embeddings()(ids)
        )

    def _overlay_embeddings(self, ids, embeddings, *, cpu_ids=None):
        result = embeddings.clone()
        if cpu_ids is not None:
            lookup = {token: i for i, token in enumerate(self.token_ids.values())}
            selected = [
                (i, lookup[token]) for i, token in enumerate(cpu_ids) if token in lookup
            ]
            if selected:
                positions, kinds = zip(*selected)
                positions = ids.new_tensor(positions)
                kinds = ids.new_tensor(kinds)
                result = result.index_copy(
                    0, positions, self.special_embeddings(kinds).to(result)
                )
            return result
        for i, token_id in enumerate(self.token_ids.values()):
            mask = ids == token_id
            result = torch.where(
                mask[..., None], self.special_embeddings.weight[i].to(result), result
            )
        return result

    def branch_embeddings(self, branch):
        return self._overlay_embeddings(
            branch.ids, branch.embeds, cpu_ids=branch.cpu_ids
        )

    def logits(self, hidden):
        output = self.root_model.get_output_embeddings()(hidden)
        if self.lm_head_mode == "unified":
            return output[..., : self.vocab_size]
        output = output[..., : self.vocab_size].clone()
        special = self.special_readout(hidden.to(self.special_readout.weight.dtype)).to(
            output
        )
        for i, token_id in enumerate(self.token_ids.values()):
            output[..., token_id] = special[..., i]
        return output

    def _encode_scene_memories(self, spatial, reference):
        """Encode one scene once and reuse its spatial states during decoding."""
        xyz, features = (spatial["memory_xyz"], spatial["memory_features"])
        block_cu, scene_cu = (
            spatial["block_cu_seqlens"],
            spatial["scene_block_cu_seqlens"],
        )
        if not len(xyz):
            raise ValueError("Spatial memory must not be empty")
        metadata = spatial.get("_cpu_metadata")
        points = (
            tuple(metadata["block_bounds"])
            if metadata is not None
            else offsets(block_cu, len(xyz), "points")
        )
        scenes = (
            tuple(metadata["scene_bounds"])
            if metadata is not None
            else offsets(scene_cu, len(points) - 1, "scenes")
        )
        if len(scenes) != 2:
            raise ValueError("Only one scene is supported")
        result = {}
        for scene in range(len(scenes) - 1):
            initial = self.ego._initial_memory(
                xyz, features, points, *scenes[scene : scene + 2]
            )
            memory, positions, cu = initial
            bounds = [0]
            for block_index in range(*scenes[scene : scene + 2]):
                bounds.append(
                    bounds[-1] + points[block_index + 1] - points[block_index] + 1
                )
            encoded = self.ego.backbone.encode_memory(
                memory.to(reference),
                memory_xyz=positions,
                block_cu_seqlens=cu,
                block_bounds=tuple(bounds),
            )
            block = []
            for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                if index == 0:
                    block.append(
                        (
                            encoded[0].new_zeros(3, dtype=torch.float32),
                            torch.eye(3, device=reference.device, dtype=torch.float32),
                        )
                    )
                else:
                    block.append(
                        self.ego.block_pose_head(encoded[0][0, start + 1 : end])
                    )
            point_count = points[scenes[scene + 1]] - points[scenes[scene]]
            result[scene] = (encoded, block, point_count)
        return result

    def _predict_memory_points(self, spatial, *, raw=False):
        if self.active_point_head is None:
            return {}
        block_bounds = spatial["block_cu_seqlens"].detach().cpu().tolist()
        scenes = spatial["scene_block_cu_seqlens"].detach().cpu().tolist()
        result = {}
        for scene, (start, end) in enumerate(zip(scenes[:-1], scenes[1:])):
            memory = spatial["_scene_cache"][scene][0][0][0]
            chunks, offset = ([], 0)
            for block in range(start, end):
                count = block_bounds[block + 1] - block_bounds[block]
                chunks.append(memory[offset + 1 : offset + count + 1])
                offset += count + 1
            points = torch.cat(chunks) if chunks else memory[:0]
            result[scene] = self.active_point_head(points)
        if self.model_config.point_coordinate_frame == "block_local" and (not raw):
            from .point_head import transform_memory_points

            result = transform_memory_points(
                result,
                {scene: entry[1] for scene, entry in spatial["_scene_cache"].items()},
                spatial["block_cu_seqlens"],
                spatial["scene_block_cu_seqlens"],
            )
        return result

    def _decode_value_features(self, hidden, kind):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if kind == "center":
                features = self.center_head[:-1](hidden.float())
                return (self.center_head[-1](features), features)
            if kind == "viewpoint":
                features = self.viewpoint_head[:-1](hidden.float())
                return self.viewpoint_head[-1](features), features
            xyz, rotation = self.ego.goal_head(hidden[None])
            return (torch.cat((xyz, matrix_to_quaternion(rotation))), hidden.float())

    def _decode_value(self, hidden, kind):
        return self._decode_value_features(hidden, kind)[0]

    def feedback_embedding(self, kind, value, head_hidden):
        if kind != "center":
            raise ValueError("vector feedback currently supports center tokens only")
        if self.feedback_mode == "pe":
            xyz = value[:3].detach().float()
            return self.prompt_embedding.position_encoder(xyz[None])[0]
        if self.feedback_mode == "hidden_state":
            with torch.autocast(device_type=head_hidden.device.type, enabled=False):
                return self.hidden_feedback(head_hidden.detach().float())
        raise ValueError("text feedback has no vector embedding")

    def format_text_feedback(self, kind, value):
        """Serialize metric XYZ in the first-block frame."""
        return format_feedback(kind, value.detach().float(), self.feedback_policy)

    def decode_trajectory(self, hidden, condition):
        condition = condition.detach().float()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            progress = self.ego.trajectory_progress.to(hidden.device)
            features = self.ego.trajectory_norm(hidden.float())
            if self.trajectory_flow:
                return self.trajectory_head.sample(features, condition)
            start, goal = (condition[0, :3], condition[1, :3])
            interior = self.trajectory_head(
                torch.cat(
                    (
                        features,
                        start.expand(78, -1),
                        goal.expand(78, -1),
                        (goal - start).expand(78, -1),
                        progress,
                    ),
                    dim=-1,
                )
            )
        world = torch.cat((condition[:1, :3], interior, condition[1:2, :3]))
        return world

    def __init__(self, text_model, *, root_model, tokenizer, model_config, protocol):
        super().__init__()
        self.model_config = model_config
        config = model_config
        task = protocol.get("block_task_config", {})
        self.ego = LingBotNaviEgoBlockModel(
            text_model,
            model_config=config,
            task_config=BlockConfig(
                hidden_dim=task.get("hidden_dim", 384),
                position_gate_init=task.get("position_gate_init", 0.1),
                spatial_full=config.spatial_tuning_mode == "full",
            ),
        )
        del self.ego._modules["text_model"]
        object.__setattr__(self.ego, "text_model", text_model)
        self.trajectory_flow = protocol.get("trajectory_flow") == TRAJECTORY_FLOW_PROTOCOL
        device = next(text_model.parameters()).device
        if self.trajectory_flow:
            from .trajectory_flow import FlowTrajectoryHead
            self.ego.trajectory_head = FlowTrajectoryHead(config.qwen_hidden_dim).to(device)
        else:
            self.ego.trajectory_head[0] = nn.Linear(
                config.qwen_hidden_dim + 10, self.ego.trajectory_head[0].out_features
            ).to(device)
        object.__setattr__(self, "_root_model", root_model)
        object.__setattr__(self, "tokenizer", tokenizer)
        self.feedback_mode = normalize_feedback_mode(protocol["feedback_mode"])
        decimals = protocol.get("feedback_policy", {})
        self.feedback_policy = FeedbackPolicy(
            decimals.get("xyz_decimals", 2), decimals.get("quaternion_decimals", 4)
        )
        self.lm_head_mode = protocol["lm_head_mode"]
        self.max_context_tokens = int(protocol["max_context_tokens"])
        self.token_ids = {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in (*SPATIAL_SPECIAL_TOKENS, VIEWPOINT_TOKEN) if token in protocol["token_ids"]
        }
        self.vocab_size = len(tokenizer)
        hidden = config.qwen_hidden_dim
        self.special_embeddings = nn.Embedding(len(self.token_ids), hidden)
        self.special_readout = (
            nn.Linear(hidden, len(self.token_ids), bias=False)
            if self.lm_head_mode == "split"
            else None
        )
        if self.lm_head_mode == "unified":
            head = root_model.get_output_embeddings()
            if head.weight is root_model.get_input_embeddings().weight:
                head.weight = nn.Parameter(head.weight.detach().clone())
            root_model.config.tie_word_embeddings = False
        self.center_head = nn.Sequential(
            nn.Linear(hidden, 384), nn.GELU(), nn.Linear(384, 3)
        )
        if VIEWPOINT_TOKEN in self.token_ids:
            self.viewpoint_head = nn.Sequential(nn.Linear(hidden, 384), nn.GELU(), nn.Linear(384, 5))
        from .point_head import MemoryPointHead

        self.point_head = self.global_point_head = None
        if config.point_head_enabled:
            if config.point_coordinate_frame == "first_block":
                self.global_point_head = MemoryPointHead(hidden)
            else:
                self.point_head = MemoryPointHead(hidden)
        if self.feedback_mode == "hidden_state":
            width = self.center_head[-1].in_features
            self.hidden_feedback = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, hidden)
            )
        self._block_outputs = {}

    def _run_branches(self, branches, spatial, scene_cache=None):
        if len(branches) != 1:
            raise ValueError("Only one query is supported")
        branch = branches[0]
        cache = spatial.get("_scene_cache")
        if cache is None:
            cache = self._encode_scene_memories(spatial, branch.embeds)
            spatial["_scene_cache"] = cache
        if scene_cache is not None:
            scene_cache["scenes"] = cache
        encoded, blocks, count = cache[0]
        if branch.cpu_ids is None:
            branch.cpu_ids = tuple(branch.ids.cpu().tolist())
        span = image_bounds(
            branch.cpu_ids,
            vision_start=self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end=self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            image_token=self.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )
        branch.cache["image_end"] = span[1]
        mask = torch.zeros_like(branch.ids, dtype=torch.bool)
        mask[span[0] : span[1]] = True
        memory, hidden = self.ego.backbone.forward_prompt(
            encoded,
            self.branch_embeddings(branch)[None],
            position_ids=branch.positions[:, None],
            current_image_mask=mask,
            image_bounds=span,
            trajectory_mask=branch.ids.eq(-1),
            memory_point_count=count,
        )
        self._block_outputs[id(branch)] = blocks
        return {0: hidden[0]}, {0: memory[0]}
