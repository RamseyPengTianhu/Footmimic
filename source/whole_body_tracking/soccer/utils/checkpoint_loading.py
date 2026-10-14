"""Checkpoint helpers for resuming across envs with extra observation terms."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner as BaseOnPolicyRunner


S2_CURRICULUM_INFO_KEY = "s2_curriculum_state"
_S2_CURRICULUM_SCALAR_NAMES = (
    "s2_curriculum_level",
    "_s2_curriculum_episode_count",
    "_s2_curriculum_fall_count",
    "_s2_curriculum_contact_success_sum",
    "_s2_curriculum_side_success_sum",
    "_s2_curriculum_side_attempt_count",
    "_s2_curriculum_first_release_quality_sum",
    "_s2_curriculum_first_release_quality_count",
    "_s2_curriculum_sequence_success_sum",
    "_s2_curriculum_sequence_episode_count",
    "_s2_curriculum_last_contact_success_rate",
    "_s2_curriculum_last_correct_side_rate",
    "_s2_curriculum_last_first_release_quality_rate",
    "_s2_curriculum_last_sequence_completion_rate",
    "_s2_curriculum_last_fall_rate",
    "_s2_curriculum_last_event_coverage",
    "_s2_curriculum_last_evaluation_count",
    "_s2_curriculum_best_contact_success_rate",
    "_s2_curriculum_pass_streak",
    "_s2_curriculum_plateau_streak",
    "_s2_curriculum_last_evaluated_level",
    "_s2_hard_replay_enabled",
)
_S2_CURRICULUM_TENSOR_NAMES = (
    "_s2_event_attempt_count",
    "_s2_event_success_count",
    "_s2_audit_event_attempt_count",
    "_s2_audit_event_success_count",
    "_s2_curriculum_level_entry_iteration",
)
_S2_CURRICULUM_INTEGER_NAMES = {
    "s2_curriculum_level",
    "_s2_curriculum_pass_streak",
    "_s2_curriculum_plateau_streak",
    "_s2_curriculum_last_evaluated_level",
    "_s2_hard_replay_enabled",
}


def _runner_env(target):
    return getattr(target, "env", target)


def _runner_base_env(target):
    env = _runner_env(target)
    return getattr(env, "unwrapped", env)


def _s2_curriculum_command(target, *, require_active_term: bool = True):
    base_env = _runner_base_env(target)
    if require_active_term:
        curriculum_manager = getattr(base_env, "curriculum_manager", None)
        active_terms = tuple(getattr(curriculum_manager, "active_terms", ()))
        if "s2_contact_levels" not in active_terms:
            return None
    try:
        command = base_env.command_manager.get_term("motion")
    except (AttributeError, KeyError):
        return None
    if not isinstance(getattr(command, "s2_curriculum_level", None), torch.Tensor):
        return None
    return command


def _s2_curriculum_signature(command) -> tuple:
    """Return the sampling/promotion contract that gives each level meaning."""
    levels = tuple(
        tuple((int(count), float(probability)) for count, probability in level)
        for level in getattr(command.cfg, "dribble_cg_curriculum_levels", ())
    )
    required_contacts = tuple(
        int(count)
        for count in getattr(
            command.cfg, "dribble_cg_curriculum_required_contacts", ()
        )
    )
    terminal_progress_levels = tuple(
        int(level)
        for level in getattr(command.cfg, "dribble_cg_terminal_progress_levels", ())
    )
    lookahead_levels = tuple(
        int(level)
        for level in getattr(command.cfg, "dribble_cg_curriculum_lookahead_levels", ())
    )
    return (
        levels,
        required_contacts,
        terminal_progress_levels,
        lookahead_levels,
        str(getattr(command.cfg, "dribble_cg_terminal_transition_mode", "arrival_radius")),
        bool(getattr(command.cfg, "dribble_cg_episode_fixed_contact_targets", False)),
    )


def set_s2_training_iteration(
    target,
    iteration: int,
    *,
    initialize_current_level: bool = False,
) -> bool:
    """Expose the RSL-RL learning iteration to the environment curriculum.

    ``initialize_current_level`` is reserved for a fresh S2 stage loaded from
    a checkpoint without S2 curriculum state. Legacy S2 checkpoints keep -1
    history entries because their earlier promotion iterations are unknowable.
    """
    command = _s2_curriculum_command(target)
    if command is None:
        return False
    resolved_iteration = int(iteration)
    value = getattr(command, "_s2_training_iteration", None)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value.fill_(resolved_iteration)
    else:
        command._s2_training_iteration = torch.tensor(
            resolved_iteration, dtype=torch.long, device=command.device
        )
    if initialize_current_level and resolved_iteration >= 0:
        history = getattr(command, "_s2_curriculum_level_entry_iteration", None)
        level = int(command.s2_curriculum_level.item())
        if (
            isinstance(history, torch.Tensor)
            and history.ndim == 1
            and 0 <= level < history.shape[0]
        ):
            history[level] = resolved_iteration
    return True


def capture_s2_curriculum_state(target) -> dict[str, Any] | None:
    """Capture global S2 curriculum and hard-event replay state."""
    command = _s2_curriculum_command(target)
    if command is None:
        return None
    state: dict[str, Any] = {
        "version": 5,
        "curriculum_signature": _s2_curriculum_signature(command),
    }
    for name in _S2_CURRICULUM_SCALAR_NAMES:
        value = getattr(command, name, None)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        scalar = value.detach().cpu().item()
        state[name] = int(scalar) if name in _S2_CURRICULUM_INTEGER_NAMES else float(scalar)
    for name in _S2_CURRICULUM_TENSOR_NAMES:
        value = getattr(command, name, None)
        if isinstance(value, torch.Tensor):
            state[name] = value.detach().cpu().clone()
    return state


def _reset_runner_env(target) -> None:
    env = _runner_env(target)
    reset = getattr(env, "reset", None)
    if callable(reset):
        reset()
        return
    base_reset = getattr(_runner_base_env(target), "reset", None)
    if callable(base_reset):
        base_reset()


def set_s2_curriculum_level(
    target,
    level: int,
    *,
    reset_env: bool = True,
    clear_statistics: bool = True,
) -> int:
    """Set an S2 level for playback or recovery and resample aligned episodes."""
    command = _s2_curriculum_command(target)
    if command is None:
        raise RuntimeError("The active environment does not expose the S2 contact curriculum.")
    configured_levels = tuple(getattr(command.cfg, "dribble_cg_curriculum_levels", ()))
    max_level = max(0, len(configured_levels) - 1)
    resolved_level = min(max(0, int(level)), max_level)
    command.s2_curriculum_level.fill_(resolved_level)
    episode_level = getattr(command, "s2_episode_curriculum_level", None)
    if isinstance(episode_level, torch.Tensor):
        episode_level.fill_(resolved_level)
    if clear_statistics:
        for name in _S2_CURRICULUM_SCALAR_NAMES:
            if name == "s2_curriculum_level":
                continue
            value = getattr(command, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()
        for name in _S2_CURRICULUM_TENSOR_NAMES:
            if name == "_s2_curriculum_level_entry_iteration":
                continue
            value = getattr(command, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()
    if reset_env:
        _reset_runner_env(target)
    return resolved_level


def restore_s2_curriculum_state(
    target,
    checkpoint: dict[str, Any],
    *,
    reset_env: bool = True,
) -> bool:
    """Restore S2 curriculum state stored under the checkpoint ``infos`` field."""
    command = _s2_curriculum_command(target)
    if command is None:
        return False
    infos = checkpoint.get("infos")
    state = infos.get(S2_CURRICULUM_INFO_KEY) if isinstance(infos, dict) else None
    if not isinstance(state, dict):
        print("[INFO] S2 curriculum: checkpoint has no saved state; starting at level 0.")
        return False

    current_signature = _s2_curriculum_signature(command)
    if state.get("curriculum_signature") != current_signature:
        # A saved numeric level is not portable across schedules. Retain the
        # policy/optimizer state, but restart the active schedule from its
        # first level.  The clean route has one Stage-2 schedule, so a
        # signature mismatch must never jump into its middle.
        set_s2_curriculum_level(
            target,
            0,
            reset_env=reset_env,
            clear_statistics=True,
        )
        print(
            "[INFO] S2 curriculum: checkpoint schedule differs from the active "
            "schedule; reset curriculum to level 0."
        )
        return False

    history = getattr(command, "_s2_curriculum_level_entry_iteration", None)
    if int(state.get("version", 0)) < 4 and isinstance(history, torch.Tensor):
        # Earlier checkpoints contain the current level but no trustworthy
        # promotion timestamps. Preserve that distinction explicitly.
        history.fill_(-1)

    configured_levels = tuple(getattr(command.cfg, "dribble_cg_curriculum_levels", ()))
    max_level = max(0, len(configured_levels) - 1)
    restored_level = min(max(0, int(state.get("s2_curriculum_level", 0))), max_level)
    for name in _S2_CURRICULUM_SCALAR_NAMES:
        if name not in state:
            continue
        saved_value = restored_level if name == "s2_curriculum_level" else state[name]
        current = getattr(command, name, None)
        dtype = torch.long if name in _S2_CURRICULUM_INTEGER_NAMES else torch.float32
        if isinstance(current, torch.Tensor) and current.numel() == 1:
            current.fill_(saved_value)
        else:
            setattr(command, name, torch.tensor(saved_value, dtype=dtype, device=command.device))
    for name in _S2_CURRICULUM_TENSOR_NAMES:
        saved_value = state.get(name)
        current = getattr(command, name, None)
        if isinstance(saved_value, torch.Tensor) and isinstance(current, torch.Tensor):
            if saved_value.shape == current.shape:
                current.copy_(saved_value.to(device=current.device, dtype=current.dtype))
            else:
                print(
                    f"[WARN] S2 curriculum: ignored {name} with shape "
                    f"{tuple(saved_value.shape)}; expected {tuple(current.shape)}."
                )
    episode_level = getattr(command, "s2_episode_curriculum_level", None)
    if isinstance(episode_level, torch.Tensor):
        episode_level.fill_(restored_level)
    if reset_env:
        # The environment was initially sampled before the checkpoint load at
        # level 0. Reset once so robot, ball, command, and recurrent episode
        # boundary all begin from the restored distribution immediately.
        _reset_runner_env(target)
    print(f"[INFO] S2 curriculum: restored level {restored_level} from checkpoint.")
    return True


def _get_alg_policy(runner):
    """Resolve policy module across rsl_rl versions (``alg.policy`` or legacy ``alg.actor_critic``)."""
    alg = runner.alg
    if hasattr(alg, "policy"):
        return alg.policy
    if hasattr(alg, "actor_critic"):
        return alg.actor_critic
    raise AttributeError(
        f"Unsupported algorithm type {type(alg)!r}: expected ``policy`` or ``actor_critic`` attribute."
    )


def _expand_2d_input_weights(old: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Append zero columns when input features grew (obs terms appended at end)."""
    if old.shape == cur.shape:
        return old
    if old.ndim != 2 or cur.ndim != 2 or old.shape[0] != cur.shape[0] or cur.shape[1] <= old.shape[1]:
        raise ValueError(f"Cannot expand weight {old.shape} -> {cur.shape}")
    pad = cur.shape[1] - old.shape[1]
    zeros = torch.zeros(old.shape[0], pad, dtype=old.dtype, device=old.device)
    return torch.cat([old, zeros], dim=1)


def _expand_1d_tail(old: torch.Tensor, cur: torch.Tensor, *, fill: float) -> torch.Tensor:
    if old.shape == cur.shape:
        return old
    if old.ndim != 1 or cur.ndim != 1 or cur.shape[0] <= old.shape[0]:
        raise ValueError(f"Cannot expand vector {old.shape} -> {cur.shape}")
    pad = cur.shape[0] - old.shape[0]
    tail = torch.full((pad,), fill, dtype=old.dtype, device=old.device)
    return torch.cat([old, tail], dim=0)


def expand_state_dict_for_obs_growth(
    checkpoint_sd: dict[str, torch.Tensor],
    current_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Pad checkpoint weights along the input-feature axis for new trailing obs dims."""
    expanded: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            expanded[key] = cur.clone()
            notes.append(f"keep init: {key}")
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            expanded[key] = old
            continue
        if old.ndim == 2 and cur.ndim == 2:
            expanded[key] = _expand_2d_input_weights(old, cur)
            notes.append(f"{key}: {tuple(old.shape)} -> {tuple(expanded[key].shape)}")
            continue
        raise ValueError(
            f"Incompatible checkpoint tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )
    return expanded, notes


def expand_obs_normalizer_state_dict(
    checkpoint_sd: dict[str, torch.Tensor],
    current_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extend running mean/var for appended observation dimensions.

    Handles both flat ``(N,)`` and row-vector ``(1, N)`` normalizer shapes.
    """
    expanded: dict[str, torch.Tensor] = {}
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            expanded[key] = cur.clone()
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            expanded[key] = old
            continue

        fill = 1.0 if "var" in key else 0.0

        # flat 1-D: (N,) -> (N+k,)
        if old.ndim == 1 and cur.ndim == 1:
            expanded[key] = _expand_1d_tail(old, cur, fill=fill)
            continue

        # row-vector 2-D: (1, N) -> (1, N+k)
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == 1 and cur.shape[0] == 1:
            pad = cur.shape[1] - old.shape[1]
            if pad <= 0:
                raise ValueError(
                    f"Incompatible normalizer tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
                )
            tail = torch.full((1, pad), fill, dtype=old.dtype, device=old.device)
            expanded[key] = torch.cat([old, tail], dim=1)
            continue

        raise ValueError(
            f"Incompatible normalizer tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )
    return expanded


def _state_dict_needs_obs_expand(checkpoint_sd: dict[str, torch.Tensor], current_sd: dict[str, torch.Tensor]) -> bool:
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            continue
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == cur.shape[0] and cur.shape[1] > old.shape[1]:
            return True
        return False
    return False


def _direct_upper_latent_layout(runner) -> tuple[list[int], int] | None:
    """Return old joint-action rows and new action size for a latent-action task.

    The direct upper-body interface keeps the original lower-body action order,
    then appends PCA latent coordinates.  It is therefore possible to preserve
    the trained lower-body output rows while intentionally reinitializing only
    the new latent rows.
    """
    env = getattr(runner, "env", None)
    base_env = getattr(env, "unwrapped", env)
    try:
        action_term = base_env.action_manager.get_term("joint_pos")
    except (AttributeError, KeyError):
        return None
    if not getattr(action_term, "uses_direct_upper_body_latent", False):
        return None
    lower_ids = getattr(action_term, "_lower_action_ids", None)
    if not isinstance(lower_ids, torch.Tensor):
        return None
    return [int(index) for index in lower_ids.detach().cpu().tolist()], int(action_term.action_dim)


def _migrate_direct_latent_action_output(
    old: torch.Tensor,
    cur: torch.Tensor,
    *,
    key: str,
    lower_action_ids: list[int],
) -> torch.Tensor:
    """Warm-start a 29-D joint-action head into a 21-D latent-action head.

    Lower-body action rows are copied exactly.  The six new upper-body latent
    rows are initialized at zero mean (and conservative exploration variance),
    which decodes to the current reference pose.  A fixed linear conversion of
    old arm targets is deliberately avoided: those targets were absolute and
    reference-clamped, whereas the new coordinates are reference-relative.
    """
    if old.ndim != cur.ndim or old.shape[0] <= max(lower_action_ids, default=-1):
        raise ValueError(f"Cannot migrate direct-latent action tensor {key}: {old.shape} -> {cur.shape}")
    if old.ndim == 2 and old.shape[1] != cur.shape[1]:
        raise ValueError(f"Cannot migrate direct-latent action tensor {key}: {old.shape} -> {cur.shape}")

    migrated = cur.clone()
    lower_count = len(lower_action_ids)
    migrated[:lower_count] = old[lower_action_ids]
    if old.ndim == 1:
        # ``std`` is the only 1-D action parameter.  Start the new latent
        # coordinates with moderate exploration so tanh does not saturate.
        migrated[lower_count:] = 0.5
    else:
        # Actor-output weights and biases for latent rows must decode to the
        # reference pose at the first step of fine-tuning.
        migrated[lower_count:] = 0.0
    return migrated


def _maybe_expand_normalizer_entry(loaded_dict: dict[str, Any], key: str, current_sd: dict[str, torch.Tensor]) -> bool:
    if key not in loaded_dict:
        return False
    ckpt_sd = loaded_dict[key]
    if not _state_dict_needs_obs_expand(ckpt_sd, current_sd):
        return False
    loaded_dict[key] = expand_obs_normalizer_state_dict(ckpt_sd, current_sd)
    return True


def prepare_loaded_dict_for_obs_expand(loaded_dict: dict[str, Any], runner) -> bool:
    """Adapt smaller-observation and legacy joint-action checkpoints in place."""
    policy = _get_alg_policy(runner)
    model_key = "model_state_dict"
    if model_key not in loaded_dict:
        return False

    ckpt_sd = loaded_dict[model_key]
    current_sd = policy.state_dict()
    layout = _direct_upper_latent_layout(runner)
    old_action_dim = int(ckpt_sd["std"].numel()) if isinstance(ckpt_sd.get("std"), torch.Tensor) else None
    new_action_dim = int(current_sd["std"].numel()) if isinstance(current_sd.get("std"), torch.Tensor) else None

    transformed_sd: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    changed = False
    for key, cur in current_sd.items():
        if key not in ckpt_sd:
            transformed_sd[key] = cur.clone()
            notes.append(f"keep init: {key}")
            changed = True
            continue

        old = ckpt_sd[key]
        if old.shape == cur.shape:
            transformed_sd[key] = old
            continue

        is_action_output = (
            layout is not None
            and old_action_dim is not None
            and new_action_dim is not None
            and old.shape[0] == old_action_dim
            and cur.shape[0] == new_action_dim
            and (old.ndim == 1 or (old.ndim == 2 and old.shape[1] == cur.shape[1]))
        )
        if is_action_output:
            lower_action_ids, _ = layout
            transformed_sd[key] = _migrate_direct_latent_action_output(
                old, cur, key=key, lower_action_ids=lower_action_ids
            )
            notes.append(f"{key}: migrated {old_action_dim}-D joint actions -> {new_action_dim}-D latent actions")
            changed = True
            continue

        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == cur.shape[0] and cur.shape[1] > old.shape[1]:
            transformed_sd[key] = _expand_2d_input_weights(old, cur)
            notes.append(f"{key}: {tuple(old.shape)} -> {tuple(cur.shape)}")
            changed = True
            continue

        raise ValueError(
            f"Incompatible checkpoint tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )

    if not changed:
        return False

    loaded_dict[model_key] = transformed_sd
    loaded_dict.pop("optimizer_state_dict", None)
    loaded_dict.pop("rnd_optimizer_state_dict", None)

    print("[INFO] Warm-start: adapted checkpoint for observation and/or action-interface growth.")
    for note in notes:
        if "->" in note:
            print(f"  {note}")

    if getattr(runner, "empirical_normalization", False):
        norm_expanded = False
        if hasattr(runner, "obs_normalizer"):
            if _maybe_expand_normalizer_entry(loaded_dict, "obs_norm_state_dict", runner.obs_normalizer.state_dict()):
                norm_expanded = True
        if hasattr(runner, "privileged_obs_normalizer"):
            if _maybe_expand_normalizer_entry(
                loaded_dict,
                "privileged_obs_norm_state_dict",
                runner.privileged_obs_normalizer.state_dict(),
            ):
                norm_expanded = True
        if norm_expanded:
            print("[INFO] Warm-start: expanded obs normalizer(s) for new command inputs.")

    return True


def _set_policy_exploration_std(runner, value: float) -> None:
    """Reset a loaded Gaussian policy to a scalar exploration standard deviation."""
    if value <= 0.0:
        raise ValueError(f"Policy exploration std must be positive; got {value}")
    policy = _get_alg_policy(runner)
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        std = parameters.get("std")
        if isinstance(std, torch.Tensor):
            std.fill_(float(value))
            return
        log_std = parameters.get("log_std")
        if isinstance(log_std, torch.Tensor):
            log_std.fill_(math.log(float(value)))
            return
    raise RuntimeError("Unable to locate policy std/log_std parameter for S2 warm-start")


def load_checkpoint_with_obs_expand(runner, path: str, **load_kwargs) -> Any:
    """Load a checkpoint, auto-expanding actor/critic inputs when obs grew (forward -> follow/control)."""
    map_location = load_kwargs.pop("map_location", getattr(runner, "device", None))
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    infos = loaded_dict.get("infos")
    checkpoint_has_s2_state = (
        isinstance(infos, dict) and S2_CURRICULUM_INFO_KEY in infos
    )
    s2_cross_stage_warm_start = bool(
        getattr(runner, "s2_warm_start_reset_optimizer", False)
        and not checkpoint_has_s2_state
    )
    if s2_cross_stage_warm_start:
        # S1 and S2 intentionally share policy/LSTM/normalizer shapes, but the
        # S1 Adam momentum and learned exploration scale are poor priors for
        # the new contact objective.
        load_kwargs["load_optimizer"] = False
    expanded = prepare_loaded_dict_for_obs_expand(loaded_dict, runner)

    if not expanded:
        result = BaseOnPolicyRunner.load(runner, path, **load_kwargs)
        restored_s2_state = restore_s2_curriculum_state(runner, loaded_dict)
        set_s2_training_iteration(
            runner,
            int(loaded_dict.get("iter", -1)),
            initialize_current_level=not restored_s2_state,
        )
        if s2_cross_stage_warm_start:
            warm_start_std = getattr(runner, "s2_warm_start_policy_std", None)
            if warm_start_std is not None:
                _set_policy_exploration_std(runner, float(warm_start_std))
            print("[INFO] S2 warm-start: reset optimizer and policy exploration std.")
        return result

    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(loaded_dict, tmp_path)
        # Fresh optimizer after input-dim growth; do not restore old Adam state.
        load_kwargs.setdefault("load_optimizer", False)
        result = BaseOnPolicyRunner.load(runner, tmp_path, **load_kwargs)
        restored_s2_state = restore_s2_curriculum_state(runner, loaded_dict)
        set_s2_training_iteration(
            runner,
            int(loaded_dict.get("iter", -1)),
            initialize_current_level=not restored_s2_state,
        )
        if s2_cross_stage_warm_start:
            warm_start_std = getattr(runner, "s2_warm_start_policy_std", None)
            if warm_start_std is not None:
                _set_policy_exploration_std(runner, float(warm_start_std))
            print("[INFO] S2 warm-start: reset optimizer and policy exploration std.")
        return result
    finally:
        os.remove(tmp_path)
