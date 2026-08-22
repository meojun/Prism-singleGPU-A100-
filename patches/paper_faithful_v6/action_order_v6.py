"""Action batching for two-phase overlap and KV migration.

The target is prepared while the source is still serving.  Only after that
readiness barrier does the source quiesce and stash its latest KV; a final
commit fetches/injects the capsules and exposes the target to routing.

This helper is independent of SGLang so the ordering contract can be tested
without importing the controller and its runtime dependencies.
"""

import copy


def _with_phase(action, phase):
    clone = copy.copy(action)
    clone.phase = phase
    return clone


def build_action_batches(
    actions,
    overlap_migration,
    kv_migration,
    activate_type,
    deactivate_type,
):
    """Return ``(batch, worker_count)`` pairs in dependency-safe order."""
    if not overlap_migration:
        return [(actions, len(actions))]

    activations = [a for a in actions if isinstance(a, activate_type)]
    middle = [
        a for a in actions
        if not isinstance(a, (activate_type, deactivate_type))
    ]
    deactivations = [a for a in actions if isinstance(a, deactivate_type)]

    # Only an activation paired with a deactivation of the same model is a
    # migration.  Demand-loading an inactive model has no source KV to fetch
    # and must keep the legacy one-shot activation path.
    deactivating_models = {a.model_name for a in deactivations}
    migrating = [a for a in activations if a.model_name in deactivating_models]
    ordinary = [a for a in activations if a.model_name not in deactivating_models]

    # The HTTP control path cannot safely process two concurrent activations
    # for the same GPU, so activation stages remain serialized.  prepare and
    # commit are shallow copies: the policy's original action stays reusable
    # for diagnostics and the phase is explicit in every action trace.
    if kv_migration and migrating:
        prepares = [_with_phase(a, "prepare") for a in migrating]
        commits = [_with_phase(a, "commit") for a in migrating]
        ordered = (
            (ordinary, 1),
            (prepares, 1),
            (middle, len(middle)),
            (deactivations, len(deactivations)),
            (commits, 1),
        )
    else:
        ordered = (
            (activations, 1),
            (middle, len(middle)),
            (deactivations, len(deactivations)),
        )
    return [(batch, workers) for batch, workers in ordered if batch]
