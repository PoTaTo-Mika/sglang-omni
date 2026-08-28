# SPDX-License-Identifier: Apache-2.0
"""Explicit factory opt-in for compiler-owned PD scheduler construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sglang_omni.config.schema import StageConfig
from sglang_omni.utils.imports import import_string

_PD_CAPABLE = "__sglang_pd_disaggregation_capable__"


def pd_disaggregation_capable(factory: Callable[..., Any]) -> Callable[..., Any]:
    setattr(factory, _PD_CAPABLE, True)
    return factory


def validate_pd_capabilities(stages: Iterable[StageConfig]) -> None:
    for stage in stages:
        if stage.pd_execution is None:
            continue
        factory = import_string(stage.factory_path)
        if not getattr(factory, _PD_CAPABLE, False):
            raise ValueError(
                f"Stage {stage.name!r} is PD-disaggregated, but factory "
                f"{stage.factory_path!r} has not declared PD support"
            )
