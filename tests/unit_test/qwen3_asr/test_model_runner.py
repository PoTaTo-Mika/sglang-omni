# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_asr.model_runner import Qwen3ASRModelRunner


def test_finalize_skips_unused_step_outputs_and_generation_bookkeeping() -> None:
    runner = object.__new__(Qwen3ASRModelRunner)
    token_ids = torch.tensor([3, 5], dtype=torch.long)
    batch_result = SimpleNamespace(
        next_token_ids=token_ids,
        can_run_cuda_graph=True,
    )
    request_data = SimpleNamespace(generation_steps=7)
    scheduler_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="r0", data=request_data)]
    )

    output = runner._finalize(
        batch_result,
        forward_batch=None,
        schedule_batch=None,
        scheduler_output=scheduler_output,
    )

    assert output.outputs == {}
    assert output.req_ids == []
    assert output.req_id_to_index == {}
    assert output.next_token_ids is token_ids
    assert output.host_token_ids is None
    assert output.can_run_cuda_graph is True
    assert request_data.generation_steps == 7
