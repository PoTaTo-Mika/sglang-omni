# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR model-runner fast paths."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.types import ModelRunnerOutput


class Qwen3ASRModelRunner(ModelRunner):
    """Skip unused per-token output objects for terminal-only ASR."""

    def _finalize(
        self,
        batch_result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        scheduler_output: Any,
        skip_rids: set[str] | None = None,
    ) -> ModelRunnerOutput:
        del forward_batch, schedule_batch, scheduler_output, skip_rids
        # note (luojiaxuan): Qwen3-ASR builds its transcript from Req.output_ids
        # only at terminalization, so per-step RequestOutput dictionaries and
        # generation-step bookkeeping have no consumer on this model path.
        host_token_ids = self._resolve_host_token_ids(batch_result)
        return ModelRunnerOutput(
            outputs={},
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
            next_token_ids=batch_result.next_token_ids,
            host_token_ids=host_token_ids,
        )


__all__ = ["Qwen3ASRModelRunner"]
