# Audar-TTS 重构前后接入 LOC 对比

本记录用同一个 `llama.cpp + NeuCodec` 后端，分别构造 T1 前与最新
`main` 的最小接入和生产增强接入。目标是回答两个不同问题：

1. 只把模型正确接进 `/v1/audio/speech`，重构能省多少代码？
2. 补齐 reference cache、并发请求管理、single-flight、失败传播和文件重校验后，
   重构能省多少代码？

## 结论

主口径排除所有测试文件和 Markdown/RST 文档，只统计接入所需代码与配置。

| 能力层级 | T1 前 | 最新 main | 节省 | 降幅 |
| --- | ---: | ---: | ---: | ---: |
| 最小接入 | 575 | 543 | 32 | 5.6% |
| 生产增强接入 | 797 | 633 | 164 | 20.6% |
| 从最小到生产的额外成本 | 222 | 90 | 132 | 59.5% |

只排除测试、但包含模型 README 的仓库变更口径如下：

| 能力层级 | T1 前 | 最新 main | 节省 |
| --- | ---: | ---: | ---: |
| 最小接入 | 609 | 577 | 32 |
| 生产增强接入 | 833 | 669 | 164 |

这解释了为什么第一次只做简单接入时几乎看不到收益：最小接入没有使用
`ReferenceEncodeService`，主要只得到声明式状态的收益。当生产增强能力一致时，
共享架构把新增能力的成本从 222 行降到 90 行。

## 固定快照

T1 前基线是 PR #807 squash commit `4e4c98a5` 的唯一父提交：

| 角色 | 分支 | 固定提交 |
| --- | --- | --- |
| T1 前基线 | `main` snapshot | `efad7215aaaf054d3597a4678e29e3370231b45a` |
| 最新基线 | `main` snapshot | `98b634332517ad2c9a88ff7f96880aae251a375c` |
| T1 前最小接入 | `luojiaxuan/audar-tts-pre-t1-minimal` | `0f75f44c65d336031ba9fdc7b3b74ca621618459` |
| T1 前生产增强 | `luojiaxuan/audar-tts-pre-t1-production` | `ec25335e9815ef53daa9c60239a1859804282270` |
| 最新最小接入 | `luojiaxuan/audar-tts-latest-minimal` | `2bf6e2cbdfbb4c006df44aa6e54d96421ad0e849` |
| 最新生产增强 | `luojiaxuan/audar-tts-latest-production` | `49cde765101bf3c4501dc1838f47ec23dbbf225b` |

最新基线固定于 2026-07-18，当时 `main` 顶部是 PR #1070。四个实验分支都从
各自基线直接分出，不把共享架构本身的历史 LOC 算进新模型接入成本。

## 能力口径

最小接入包含：

- Audar 请求校验、Arabic/English 文本和一组 5-15 秒 reference + transcript。
- 官方 prompt/speech-token 协议、采样参数和 seed。
- GGUF 下载及 `llama-cpp-python` 推理。
- NeuCodec reference encode 和 24 kHz waveform decode。
- `/v1/audio/speech` 输出和 usage 字段。

生产增强接入在此基础上增加：

- 按 item 数和 byte 数限制的 reference LRU cache。
- 包含 model、revision、encoder、config 和 input 的复合 cache key。
- 同 reference 并发请求 single-flight，含 follower 失败传播和超时。
- 不同 reference 请求可并发排队；共享 NeuCodec forward 由模型级锁串行保护。
- reference path 编码后的内容重校验，避免文件中途变化后写入旧 key。
- cache hit/miss/merged/failure/eviction 统计与周期日志。
- vocoder 无 batch wait；NeuCodec 当前接口没有实现张量级 batch decode。

两边的生产增强行为相同。T1 前版本在 Audar 模型目录内实现上述 reference
service；最新版本复用 `ReferenceEncodeService`、声明式 pipeline state 和
usage helper。

## LOC 来源

| 文件 | T1 前最小 | 最新最小 | T1 前生产 | 最新生产 |
| --- | ---: | ---: | ---: | ---: |
| `payload_types.py` | 64 | 20 | 64 | 20 |
| `stages.py` | 272 | 271 | 494 | 361 |
| `__init__.py` | 2 | 14 | 2 | 14 |
| `config.py` | 60 | 61 | 60 | 61 |
| 其余代码/配置 | 177 | 177 | 177 | 177 |
| **合计** | **575** | **543** | **797** | **633** |

声明式 state 直接省 44 行；生产版本的共享 reference 层再省 133 行。
最新架构新增 capability 声明和配置元数据 13 行，因此最终净节省 164 行。

## 复现统计

使用仓库现有统计脚本，并通过 Git pathspec 排除文档：

```bash
python3 scripts/refactor_net_deletions.py \
  --base <baseline-sha> \
  --head <integration-sha> \
  --mode direct \
  --path ':(exclude)**/*.md' \
  --path ':(exclude)**/*.rst'
```

`scripts/refactor_net_deletions.py` 会另外识别并排除 `tests/`、`test/`、
`unit_test/`、`integration_test/`、`test_*.py`、`*_test.py`、`conftest.py`
等测试路径。去掉两个文档 pathspec 即可复现“只排除测试”的第二口径。

验证结果：

| 分支 | 命令范围 | 结果 |
| --- | --- | ---: |
| T1 前最小 | `tests/unit_test/audar_tts` | 8 passed |
| T1 前生产 | `tests/unit_test/audar_tts` | 14 passed |
| 最新最小 | Audar + pipeline-state + capabilities | 55 passed |
| 最新生产 | Audar + pipeline-state + capabilities | 65 passed |

四个分支的 Audar 文件均通过 `ruff format --check` 和 `ruff check`。

## 3k 行说法的边界

[Audar-TTS-V1-Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo)
只发布 GGUF，官方示例也是 `llama.cpp`；本实验因此把 1.64B Qwen2 backbone
和 NeuCodec 实现留在 `llama-cpp-python`/`neucodec` 依赖中，不 vendoring、也不
计入 SGLang-Omni 接入 LOC。

截至 2026-07-18，Sandy 的公开 fork、上游 PR 和远端分支中没有可核验的
Audar 3k 行实现。因此不能把这里的 633 行直接解释成“同一份 3k 实现被重构
压到 633 行”。如果 3k 包含原生 PyTorch/SGLang model、GGUF weight loader、
model runner 或 continuous batching，它属于不同 backend 和能力口径，必须拿到
分支后按同一 manifest 重新比较。

当前“生产增强”仍使用单个 `llama_cpp.Llama` 和串行 AR `SimpleScheduler`。
它补齐的是 reference/vocoder 侧的生产能力，不包含 SGLang 原生 AR continuous
batching、TP 或 streaming。这个限制在重构前后完全相同，所以 LOC 对比公平，
但不能用来证明原生 SGLang 接入只需 633 行。

## Source of Truth

- 代码与轻量结果：上述四个 Git 分支和本文件。
- 模型：[`audarai/Audar-TTS-V1-Turbo`](https://huggingface.co/audarai/Audar-TTS-V1-Turbo)，官方 Hugging Face 仓库。
- 轻量 benchmark 汇总在 latest production 分支的
  `tests/benchmark/audar_tts/RESULTS.md` 和 `comparison.json`。
- 28 次请求的原始输出暂存于 H100
  `/data/jaxan/audar-results-production-equal-final`。计划上传到正确 owner 的
  Hugging Face dataset repo；当前缺少对应 credential，状态为 pending。
