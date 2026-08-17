"""Target-first migration action used by paper-faithful-v3."""

import dataclasses
import logging
import time

from sglang.multi_model.scheduling.action import ActivateAction, BaseAction, DeactivateAction

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OverlapMigrateAction(BaseAction):
    """Load the target fully, then drain/deactivate the source.

    The worker-pool HTTP handler is patched to return only after the engine's
    activation/deactivation completion event, so action ordering here is a real
    readiness barrier rather than merely ordering two HTTP submissions.
    """

    source_gpu_id: int = 0
    target_gpu_id: int = 0
    memory_pool_size: int = 0

    def execute(self, url, model_instance_state_dict, request_timeout=None):
        started = time.monotonic()
        activate = ActivateAction(
            model_name=self.model_name,
            instance_idx=self.target_gpu_id,
            memory_pool_size=self.memory_pool_size,
            gpu_id=self.target_gpu_id,
        )
        if not activate.execute(url, model_instance_state_dict, request_timeout):
            return False
        ready = time.monotonic()
        deactivate = DeactivateAction(
            model_name=self.model_name,
            instance_idx=self.source_gpu_id,
            preempt=False,
            preempt_mode="RECOMPUTE",
            evict_waiting_requests=False,
            gpu_id=self.source_gpu_id,
        )
        success = deactivate.execute(url, model_instance_state_dict, request_timeout)
        logger.info(
            "[PAPER-OVERLAP] model=%s source=%s target=%s target_ready_s=%.3f total_s=%.3f success=%s",
            self.model_name,
            self.source_gpu_id,
            self.target_gpu_id,
            ready - started,
            time.monotonic() - started,
            success,
        )
        return success
