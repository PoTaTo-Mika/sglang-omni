# 配置接口重构设计方案

# 1. 背景与范围

SGLang-Omni 当前的配置方式有以下几种：模型 Python 默认参数、Pipeline YAML、动态 dotted CLI、专用 typed CLI、环境变量和 Router launcher YAML。多种入口本身不是问题：YAML 适合保存完整部署的静态参数，CLI 适合选择配置和动态的临时覆盖，环境变量适合进程级注入。但是现在的配置逻辑主要存在三个问题：

1. **处理逻辑不统一**：同一个配置项从不同入口传入时，走的不是同一套处理代码。例如 YAML 和 CLI 都能设置同一个参数时，字段寻址、类型转换和参数校验可能需要重复实现。
2. **开发者扩展路径不明确**：开发者增加一个配置项时，不清楚它应该进入 typed schema、CLI option、`factory_args`、`runtime_overrides` 还是模型专属配置，也不知道需要同步修改哪些模块。
3. **用户修改路径不明确**：用户不清楚某个配置应该写入 YAML 的哪个层级、使用哪个 CLI flag，以及多个入口同时设置时最终哪个值生效。

综上，重构必须同时解决“实现是否统一”“开发者如何扩展”和“用户如何配置”三个问题，而不只是替换当前的 merge 实现。

## 1.1 一句话方案

**一个参数只定义一次；YAML 用来长期保存，CLI 用来临时覆盖；无论从哪里传入，都交给同一段代码解析和校验。**

重构前，同一个参数可能拥有多条处理路径：

```text
同一个参数
├── YAML typed field
├── stage_overrides
├── runtime_overrides
├── factory_args
├── dotted CLI
└── typed CLI helper
```

每条路径都可能包含自己的字段查找、类型转换、优先级和校验代码。

重构后，一个参数只有一个正式地址：

```text
一个参数
└── 一个配置路径
    ├── YAML：长期保存
    ├── CLI --set：临时覆盖
    └── Router：结构化传入
            ↓
       同一个配置处理器
            ↓
       最终只读配置
```

例如 `max_running_requests` 只定义为：

```text
pipeline.stages.thinker.runtime.sglang.max_running_requests
```

YAML 可以持久化设置：

```yaml
pipeline:
  stages:
    thinker:
      runtime:
        sglang:
          max_running_requests: 32
```

CLI 可以临时覆盖同一个路径：

```bash
sgl-omni serve \
  --config config.yaml \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=32
```

两种写法最终进入同一套类型转换、范围校验、冲突检查和优先级处理代码。

## 1.2 三个目标对应的解决办法

| 当前问题 | 重构后的规则 |
|---|---|
| YAML 和 CLI 重复实现参数处理 | 所有入口先转换成统一的配置修改，再交给同一个处理器 |
| 开发者不知道参数应该加在哪里 | 公共参数只加到公共 schema；模型参数只加到模型 extension；服务参数只加到 server config |
| 用户不知道应该从哪里修改 | 长期配置写 YAML，临时覆盖用 `--set`，host/port/log 等服务参数使用专用 CLI |

开发者新增一个公共参数时，只需要在 schema 中定义一次类型、默认值和校验，再在 runtime adapter 中消费。CLI 帮助、YAML Schema、文档和等价性测试都从这份定义生成。

用户不需要理解 `factory_args`、`runtime_overrides` 或内部 planner。用户只需要找到参数的正式路径，并决定这次修改是“写入 YAML 长期保存”还是“使用 CLI 临时覆盖”。

## 1.3 文档中的术语

后文使用三个技术名称表示上述简单流程：

- **canonical path**：一个参数唯一、正式的配置地址；
- **ConfigPatch**：一次“把某个地址设置成某个值”的修改记录，同时保存来源是 YAML、CLI 还是 Router；
- **ConfigResolver**：统一执行类型转换、优先级、冲突检查和校验的配置处理器。

这三个名称不是三套新接口，而是同一条配置处理管道的三个环节。

本文只讨论启动期配置，不改变请求协议、Stage payload、调度算法或 SGLang 内部 `ServerArgs` 的字段定义。文中“当前实现”核对的代码 revision 为 `2b45073c`；完整现状可参考 `docs/developer_reference/configuration.md`。

Tracking issue: [#1466](https://github.com/sgl-project/sglang-omni/issues/1466)

English version: [Configuration Interface Refactoring Design Proposal](./configuration_interface_refactor_proposal_en.md)

重构目标是：

1. **统一配置处理代码**：同一个用户语义只有一个 canonical config path；YAML、CLI 和 Router 全部进入同一个解析、类型转换、优先级、校验和冲突处理流程。
2. **建立唯一的开发者扩展接口**：新增公共配置只修改一次 typed schema；CLI、JSON Schema、文档、patch 类型和等价性测试从 schema 生成。模型专属参数使用明确的 extension schema，不再任意选择多个透传 dict。
3. **建立可发现、可解释的用户接口**：明确 YAML 与 CLI 的职责，提供 schema、validate、resolve、explain 和 migrate 工具，让用户可以找到合法配置路径并确认最终值来源。

配置在进入 placement/runtime 前应一次性 resolve、完整校验并冻结；旧接口通过分阶段迁移兼容，但不永久保留两套语义。

本文不要求一次性重写全部模型配置。新接口先稳定配置边界，再逐步迁移模型专属参数。

# 2. 当前配置接口与执行链路

## 2.1 配置来源

当前公开配置来源可以分成七类：

| 来源 | 当前入口 | 主要实现 |
|---|---|---|
| 模型默认 | `PipelineConfig` 子类、`Variants` | `models/*/config.py`、`config/manager.py` |
| Pipeline YAML | `sgl-omni serve --config` | `ConfigManager.from_file()` |
| YAML 紧凑覆盖 | `stage_overrides`、`runtime_overrides` | `config/manager.py`、`config/runtime.py` |
| 动态 CLI | `--stages.thinker... VALUE` | `ConfigManager.parse_extra_args()` |
| 专用 CLI | `--mem-fraction-static` 等 | `cli/serve.py` 中的 `apply_*_cli_overrides()` |
| 已配置的环境默认值 | Pipeline `env_defaults`、stage 级 `StageConfig.env` | `config/schema.py`、`pipeline/mp_runner.py`、`pipeline/stage_workers.py` |
| Router launcher | `--launcher-config`、`worker_extra_args` | `sglang_omni_router/launcher/*` |

`sgl-omni serve` 对未知 option 开启 `ignore_unknown_options`，因此 Typer 先解析 41 个专用参数，剩余 token 再由 `ConfigManager` 解析为 dotted override。两类 CLI 在不同阶段执行。

已配置的环境值具有比下文配置 layer 更窄的契约。Pipeline `env_defaults` 是 canonical 已配置默认值，`StageConfig.env` 在对应 stage 上覆盖它们。spawn 时，这些默认值只填充继承进程环境中不存在的变量，因此绝不覆盖 `os.environ`。最终的 `ResolvedProcessEnv` 是派生的进程产物，组合继承环境、已配置默认值以及 runner 拥有的 rank/device 等增量；它不是新的用户 precedence layer。库或 runtime 中直接读取 `os.environ` 的位置仍是外部输入，应在相关环境诊断中展示，而不是全部收进一个通用配置层。

## 2.2 当前合并顺序

当前主进程按以下顺序构造最终 `PipelineConfig`：

1. 有 `--config` 时用 YAML 中的 `config_cls` 构造配置类；否则通过 `model_path` 识别模型 architecture；
2. YAML `stage_overrides` 深合并到 typed `StageRuntimeConfig`；
3. `ctx.args` 中的 dotted CLI 修改 `model_dump()` 后的 dict，再重建 Pydantic model；
4. 显式 `--model-path` 覆盖 YAML 中的 `model_path`；
5. `serve.py` 依次执行 memory、TP/GPU、process、CUDA graph、compile、decode、prefill 和 generation batch helper；
6. `launch_server()` 将 HTTP/API 参数与 `PipelineConfig` 分开传入；
7. runtime preparation 构造 placement、process topology、endpoint 和 worker spec；
8. spawn 前，父进程把 `factory_args`、`runtime_overrides` 和 typed runtime mapping 合并为与签名无关的 factory args，并写入 worker spec；
9. 子进程 import factory，只注入缺失且是否适用依赖 factory 签名的默认值，然后由 factory 构造 `ServerArgs` 和 `ModelConfig`。

| 顺序 | 处理阶段 | 输入与输出 |
|---|---|---|
| 1 | 选择基础配置 | 模型 Python 默认或 Pipeline YAML → `ConfigManager` |
| 2 | 应用 YAML 紧凑覆盖 | `stage_overrides` → typed runtime |
| 3 | 应用动态 CLI | dotted CLI → 重建 `PipelineConfig` |
| 4 | 应用专用 CLI | typed CLI helpers → 原地修改 `PipelineConfig` |
| 5 | 运行时规划 | `PipelineConfig` → placement、process topology、父进程已解析的静态 factory args 和 worker spec |
| 6 | worker 构造 | worker spec → import factory → 注入缺失的签名相关默认值 → `ServerArgs` |

该顺序表面上提供了优先级，但优先级并不是一条统一规则。不同 helper 会选择“覆盖”“拒绝”“同步两个 alias”“同时写 `factory_args` 和 `runtime_overrides`”或“等到 worker 再报冲突”。

## 2.3 当前 factory 参数路径

stage factory kwargs 跨明确的父/子进程边界组合：

```text
factory_args
< runtime_overrides
< typed runtime 映射
→ worker spec 中父进程已解析的静态 factory args
+ 子进程注入缺失的 signature-dependent defaults
```

其中 `server_args_overrides` 在 `factory_args` 和 `runtime_overrides` 之间执行一层 dict merge；typed `runtime.max_seq_len` 和 `runtime.video_fps` 通过 `runtime_arg_map` 改名；typed `mem_fraction_static` 最终又写回 `server_args_overrides`。

父进程在 spawn 前完成所有不需要检查 factory 签名的 merge。`model_path`、`gpu_id` 和 `total_gpu_memory_fraction` 不是普通用户 factory args：父进程计算候选值，子进程 import factory 后，仅当签名声明该参数且已解析静态参数中尚无该值时才注入。`gpu_id`、process cumulative budget、NCCL port 和具体 IPC/socket endpoint 由 planner/runner 派生。

# 3. 当前设计的问题

## 3.1 同一语义存在多个用户入口

`mem_fraction_static` 是最完整的代表：

| 入口 | 写入位置 | 校验位置 |
|---|---|---|
| YAML typed runtime | `stage.runtime.sglang_server_args.mem_fraction_static` | Pydantic schema |
| YAML compatibility override | `runtime_overrides.<stage>.server_args_overrides.mem_fraction_static` | runtime adapter |
| 完整 stage YAML | `factory_args.server_args_overrides.mem_fraction_static` | runtime adapter |
| dotted CLI | 可写上面任意路径 | merge 后 Pydantic/runtime adapter |
| global typed CLI | `--mem-fraction-static` | `apply_mem_fraction_cli_overrides()` |
| role typed CLI | `--thinker-mem-fraction-static` 等 | 同一 helper 和 role map |

当前代码需要显式检测 typed runtime 与 `server_args_overrides` 的重复来源。这种检测只能覆盖已知冲突；新增来源时容易遗漏。

generation batch 也有同类问题。`max_running_requests` 可以来自：

- 模型 factory 默认；
- YAML `factory_args`；
- YAML `runtime_overrides`；
- dotted CLI；
- `--thinker-max-running-requests`；
- `--max-running-requests`；
- 模型 generation batch policy 的派生或校验。

用户看到的是一个功能，代码维护的是多条路径。

## 3.2 YAML 同时承担“配置实例”和“补丁”

当前 YAML 可以是：

1. 只包含 `config_cls` 和 `model_path` 的模板选择；
2. 带 `stage_overrides` 的紧凑 patch；
3. 带 `runtime_overrides` 的 factory patch；
4. 带完整 `stages` 的 resolved-like 配置。

四种写法进入不同 merge 逻辑。特别是：

- `stage_overrides` 只允许 typed `runtime`；
- `runtime_overrides` 是自由 dict；
- 完整 `stages` 整体替换模型默认列表；
- 同一 YAML 中可以混合这些形式。

因此“这个字段应该写在哪里”不能只从字段语义回答，还取决于用户选择了哪种 YAML 形态。

## 3.3 CLI 既是接口层，也是业务层

`cli/serve.py` 不仅声明 option，还负责：

- public role 到模型 stage 的解析；
- 修改 nested Pydantic object；
- 同步 `tp_size` 与 `parallelism.tp`；
- 把参数写入 `factory_args` 和 `runtime_overrides`；
- 检查 factory 是否属于支持集合；
- 执行 GPU topology probe；
- 重跑部分 Pydantic validation。

这导致新增一个 runtime 参数通常需要同时修改：

1. Typer 签名；
2. CLI helper；
3. role map；
4. runtime adapter 或 factory；
5. 测试；
6. 文档。

CLI 不再是输入 adapter，而成为第二个配置编译器。

## 3.4 原地 mutation 绕过 schema validation

当前 Pydantic models 没有启用 `validate_assignment=True`。`ConfigManager.merge_config()` 会 dump、deepcopy、重建，因此能执行完整 validation；大多数 typed CLI helper 则直接修改：

- `stage.gpu`
- `stage.tp_size`
- `stage.runtime.*`
- `stage.factory_args`
- `pipeline_config.runtime_overrides`

helper 必须自行保证原子性和校验完整性。部分路径会手动重建整个 config，部分路径只做局部范围检查。配置是否合法取决于修改它的具体 helper，而不是 config schema 本身。

## 3.5 动态 dotted CLI 缺少稳定契约

动态 CLI 把 `-` 转 `_`，按 `.` 遍历 dict/list，并只支持 bool、None、int、float 和 string scalar。它存在以下限制：

- 可以写任意内部字段，包括不应公开的兼容字段；
- stage list 同时支持数字 index 和 name，导出配置变化后 index 不稳定；
- list/dict 没有统一编码；
- path 不存在时可能抛出 `KeyError`/`IndexError`，错误格式不统一；
- 无法从 Typer help 得到可用路径；
- 内部字段重命名会直接破坏用户命令。

开放 override 本身有价值，但必须针对 public schema，而不是直接暴露 `model_dump()` 的内部结构。

## 3.6 Router 把结构化配置重新变成字符串

这里涉及两份作用不同的 YAML：

1. **Router launcher YAML**：描述启动几个完整 worker、为它们分配哪些 GPU 和端口；
2. **Pipeline YAML**：描述每个 worker 内部使用哪个 Pipeline、stage topology 和 runtime 参数。

当前 Router launcher YAML 不能结构化描述 worker 的 Pipeline 配置，只能把 worker 参数写成一段字符串：

```yaml
launcher:
  model_path: Qwen/Qwen3-Omni
  num_workers: 2
  worker_extra_args: "--config examples/configs/qwen.yaml --colocate"
```

Router 先为一个 worker 生成基础命令：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni \
  --host 127.0.0.1 \
  --port 8011
```

`worker_extra_args` 此时仍然只是普通字符串。Router 使用 `shlex.split()` 将它拆成：

```python
["--config", "examples/configs/qwen.yaml", "--colocate"]
```

再把这些参数追加到基础命令：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni \
  --host 127.0.0.1 \
  --port 8011 \
  --config examples/configs/qwen.yaml \
  --colocate
```

Router 启动该 worker 子进程时，还通过环境变量设置物理 GPU：

```bash
CUDA_VISIBLE_DEVICES=0
```

worker 启动之后，`sgl-omni serve` 才会解析 `--config`，再由 `ConfigManager` 加载第二份 Pipeline YAML。因此实际链路是：

```text
Router launcher YAML
→ 读取 worker_extra_args 字符串
→ shlex.split 拆成 worker CLI 参数
→ 启动 sgl-omni serve 子进程
→ worker 解析 --config
→ ConfigManager 加载 Pipeline YAML
```

问题不是 Router 启动 worker 本身，而是 Router 只知道一段字符串，不理解其中配置了什么。字段类型、重复 option、Pipeline 文件路径和参数兼容性都只能等 worker 启动后验证；Router schema 无法提前说明或检查 worker 的实际配置。

## 3.7 最终配置缺少 provenance

当前 debug merged config 只显示最终 `PipelineConfig`，不能回答：

- 某个值来自模型默认、YAML 还是 CLI；
- 它覆盖了哪个旧值；
- 是否通过 deprecated path 设置；
- 后续是否还会被 factory defaults 或 `ServerArgs.override()` 修改。

当多个入口冲突时，维护者只能反向阅读 `serve()` 的 helper 顺序。

# 4. 设计原则与非目标

## 4.1 三个核心目标如何落地

| 核心目标 | 设计机制 | 验收标准 |
|---|---|---|
| 统一配置处理代码 | canonical schema、`ConfigPatch`、唯一 `ConfigResolver` | YAML、`--set` 和迁移期 CLI alias 对同一字段产生相同 resolved value，且不再包含入口专属 mutation helper |
| 明确开发者扩展接口 | public typed schema、model extension schema、schema metadata 生成 CLI/文档/测试 | 新增公共字段只定义一次；新增模型字段只能进入注册过的 extension schema |
| 明确用户修改方式 | YAML/CLI 职责边界、name-keyed path、`config schema/validate/resolve/explain/migrate` | 用户可以发现合法 path、在启动前验证配置，并看到最终值及覆盖来源 |

后续所有数据结构、迁移步骤和测试都必须服务于这三个目标。若某项设计只减少内部代码、却没有降低开发者或用户的选择成本，则不视为完成本次重构。

## 4.2 设计原则

1. **一个语义，一个 canonical path**：相同运行时功能不得同时暴露 typed 和 untyped public path。
2. **输入 adapter 无业务逻辑**：YAML、CLI、Router 只产生 typed assignment/patch。
3. **一次 resolve，一次完整 validation**：进入 runtime planner 后配置冻结。
4. **用户配置与派生计划分离**：GPU logical placement 和 endpoint 分配根目录 `endpoints.base_path` 可以配置，`gpu_id`、具体 socket endpoint、NCCL port 等 runtime 值不可配置。
5. **显式覆盖，保留来源**：跨层覆盖允许，但必须可解释；同 layer、相同 specificity 的重复定义直接失败。
6. **兼容层有删除终点**：旧接口可以适配，但不能永久成为第二套 canonical schema。

## 4.3 非目标

- 不把 SGLang 的全部 `ServerArgs` 原样复制成 Omni 顶层 CLI；
- 不让 YAML 选择 ZMQ/CUDA IPC 等由 locality/placement 推导的 transport；
- 不在本次重构中改变 Stage/process/TP 的运行时语义；
- 不允许 request-time 参数进入启动期 `PipelineConfig`；
- 不要求 Router 与 worker 共享同一个顶层 schema，它们只共享 patch 表达。

# 5. 新设计总览

新设计增加一个单向配置编译阶段：

```text
Input documents
→ adapters
→ ConfigPatchSet
→ ConfigResolver
→ immutable ResolvedPipelineConfig
→ runtime planners
→ worker specs
```

| 顺序 | 处理阶段 | 职责 |
|---|---|---|
| 1 | 输入 adapter | Python defaults、YAML、CLI、兼容入口和 Router 都转换成 `ConfigPatch` |
| 2 | `ConfigPatchSet` | 保存所有修改及其来源，不执行入口专属业务逻辑 |
| 3 | `ConfigResolver` | 统一处理类型、优先级、重复定义、冲突和完整校验 |
| 4 | `ResolvedPipelineConfig` | 生成完整、只读并带 provenance 的最终用户配置 |
| 5 | Runtime planners | 只读最终配置，生成 placement、process 和 derived runtime plans |
| 6 | Worker specs | 根据最终配置和 plans 构造跨进程启动参数 |

所有 adapter 只依赖 public schema 和 path registry。`ConfigResolver` 是唯一执行 precedence、alias normalization、duplicate detection 和 merge 的组件。runtime planner 不再读取 unresolved compatibility fields。

# 6. 新的用户接口

## 6.1 YAML 是 Pipeline 的持久化声明

当前 YAML 既可以完整列出所有 stages，也可以只写基于模型默认值的修改项，但文件本身没有明确说明自己属于哪一种。V2 使用 `document_mode` 把两种用途区分开。

### `full`：完整 Pipeline

`full` 从空 topology 开始构造，不继承模型配置类的默认 stages。文件必须完整声明 entry stage、所有 stages 和构造顺序，单独拿到该文件就能确定整个 Pipeline。下面的 `example.*` factory 仅用于展示结构。

```yaml
schema_version: 2
document_mode: full

model:
  path: Qwen/Qwen3-Omni-30B-A3B-Instruct

pipeline:
  name: qwen3-omni-text
  entry_stage: preprocessing

  stage_order:
    - preprocessing
    - thinker
    - decode

  stages:
    preprocessing:
      kind: python
      factory: example.create_preprocessing
      process: pipeline
      next: thinker

    thinker:
      kind: sglang
      factory: example.create_thinker
      process: pipeline
      next: decode
      placement:
        gpus: [0]
      runtime:
        sglang:
          max_running_requests: 32

    decode:
      kind: python
      factory: example.create_decode
      process: pipeline
      terminal: true
```

`stage_order` 必须与 `stages` 中的名称一一对应，不能缺少、重复或引用不存在的 stage。`config export` 默认输出这种完整形式，适合归档和精确复现。

### `partial`：在模型模板上修改少量字段

`partial` 必须通过 `model.config` 或 variant 选择一个模型模板。resolver 先创建模板的默认 Pipeline，再应用 YAML 中写出的修改；没有写出的 stage 和字段保持模型默认值。

```yaml
schema_version: 2
document_mode: partial

model:
  path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  config: Qwen3OmniSpeechColocatedPipelineConfig

pipeline:
  stages:
    thinker:
      runtime:
        sglang:
          max_running_requests: 32
```

这个例子的含义只有三步：

1. 创建 `Qwen3OmniSpeechColocatedPipelineConfig` 默认 Pipeline；
2. 找到名为 `thinker` 的 stage；
3. 把 `max_running_requests` 改成 32。

`partial` 可以省略不修改的内容，但只能修改模板中已经存在的 stage，不能创建或删除 stage。需要改变完整 topology 时，应使用 `full`。

### 为什么 stages 改成按名称保存

当前配置把 stages 保存为 list，动态 CLI 可能需要使用不稳定的数字位置：

```text
stages.4.runtime...
```

V2 将 stages 保存为 name-keyed mapping：

```text
pipeline.stages.thinker.runtime...
```

stage 名成为稳定身份，YAML 和 CLI 可以使用同一个路径，不再依赖 list index。mapping 本身不负责表达构造顺序，因此 `full` document 额外使用 `stage_order`；`partial` 直接继承模板已有的顺序。

当前 `stage_overrides` 之所以存在，是因为用户需要按 stage 名修改默认 stage list。V2 的 `partial.pipeline.stages.<name>` 本身就是 canonical 修改路径，因此不再需要单独设计一套 `stage_overrides` merge 逻辑。

## 6.2 CLI 只保留启动参数和通用 patch

steady-state `serve` 接口分成三组：

### 配置选择

```text
--config FILE
```

或：

```text
--model-path MODEL [--variant text|speech|...]
```

两种模式互斥。`--model-path` 是“从模型默认构造配置”的 shorthand；当 `--config` 已包含 `model.path` 时，不能再覆盖它。

### 服务进程参数

以下值不属于 Pipeline，继续作为 CLI option：

```text
--host
--port
--model-name
--log-level
--allowed-local-media-path
--allowed-media-domain
--tts-batch-max-items
--enable-realtime
```

它们由 `ServerLaunchConfig` 单独接收，不写入 `PipelineConfig`。如未来需要保存服务配置，应引入独立 deployment document，而不是塞入 Pipeline YAML。

### Pipeline 临时覆盖

统一使用可重复的：

```text
--set PATH=VALUE
```

例如：

```bash
sgl-omni serve \
  --config qwen.yaml \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=32 \
  --set pipeline.stages.thinker.placement.gpus='[0,1]' \
  --set pipeline.stages.thinker.placement.tp=2
```

value 使用 YAML scalar/flow value 解析，因此 bool、null、number 和原子 list value 具有同一类型规则。`--set` 只能定位 configurable leaf；list 整体视为一个 leaf，不能按 index patch。通用 `--set` 不能替换 mapping/subtree；subtree replacement 只允许写在 `full` YAML 中。path 必须存在于 public schema；不能写 runtime-derived 或 deprecated internal path。

通用 stage 参数还支持 selector-based broadcast patch：

```yaml
pipeline:
  stage_defaults:
    - select:
        capability: sglang
      runtime:
        sglang:
          max_running_requests: 32
    - select:
        roles: [thinker, generation]
        exclude: [talker_ar]
      placement:
        tp: 2
```

selector 支持 `capability`、`roles`、`stages` 和 `exclude`；给出的正向 selector 必须全部匹配，`exclude` 再移除匹配项。resolver 在 duplicate 检查和 validation 前，把 broadcast 展开为具体 stage leaf patch。resolved storage 仍只保存 `pipeline.stages.<stage>...`，`ResolvedPipelineConfig` 中不保留 broadcast object。同一个 source 内 specificity 为 `capability broadcast < role/group selector < concrete stage`；不同 source 之间始终由 source layer 主导，specificity 不能越层。相同 specificity 写同一个展开 leaf 仍报错。

对应 CLI 为：

```bash
sgl-omni serve \
  --config qwen.yaml \
  --set-for capability=sglang runtime.sglang.max_running_requests=32 \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=48
```

这里同一 CLI source 中的具体 stage assignment 覆盖 capability broadcast。`config explain` 展示每次 selector expansion 和被遮蔽的值；full export 输出展开后的 stage-local value，不输出 selector。

## 6.3 专用 CLI 的最终定位

现有 runtime 专用 CLI 在迁移期保留，但不再执行 helper。每个 option 只声明一个 canonical path：

```python
CliAlias(
    option="--thinker-max-running-requests",
    path="pipeline.stages.${role:thinker}.runtime.sglang.max_running_requests",
)
```

解析后产生普通 patch。迁移 alias 在 CLI source 内具有明确 specificity：broadcast/global alias < role alias < 显式具体 `--set`。role alias 覆盖 broadcast alias 是兼容行为，不是 duplicate；被遮蔽的 broadcast value 保留在 provenance 中。相同 specificity 的两个 assignment 写同一 canonical leaf 仍报错。source layer precedence 始终优先于此规则。

V2 稳定后，runtime 专用 CLI 分批删除，只保留 `--set` 和服务进程参数。这样最终接口不会长期维护“YAML 字段 + 专用 CLI 字段”两份定义。若项目决定长期保留极少数高频 alias，它们必须由 schema metadata 自动生成 help、类型和 patch，不允许拥有独立业务代码。

## 6.4 配置工具

增加：

```bash
sgl-omni config schema
sgl-omni config validate --config FILE
sgl-omni config resolve --config FILE [--set ...]
sgl-omni config explain --config FILE PATH [--set ...]
sgl-omni config export --config FILE --format full
sgl-omni config migrate --input v1.yaml --output v2.yaml
```

`resolve` 输出冻结前的完整 canonical Pipeline；`explain` 输出最终值、source chain 和是否使用 deprecated adapter。例如：

```text
pipeline.stages.thinker.runtime.sglang.max_running_requests = 32

1. model-default:Qwen3OmniSpeechPipelineConfig  64
2. file:qwen.yaml:41                           48
3. cli:--set[0]                                32
```

## 6.5 用户配置决策

用户只需要按以下规则选择入口：

1. 需要保存、review 或复现的 Pipeline 改动，写入 YAML canonical path；
2. 只在本次启动临时覆盖 Pipeline 字段，使用 `--set` 写同一个 canonical path；
3. 修改 host、port、日志或 API policy，使用服务进程 CLI；
4. 不知道字段路径时，先运行 `config schema`；不确定最终值时，运行 `config resolve` 或 `config explain`；
5. 不直接设置 `factory_args`、`runtime_overrides` 或 runtime-derived fields。

因此文档和错误信息不再回答“这个值在五种入口中应该选哪个”，而只需要回答“这个值属于哪个 canonical path，以及本次改动应持久化还是临时覆盖”。

# 7. 核心内部数据结构

## 7.1 `ConfigPatch`

```python
class ConfigSource(BaseModel):
    kind: Literal["model_default", "file", "cli", "router", "compat"]
    location: str
    deprecated: bool = False


class StageDeclaration(BaseModel):
    name: str
    kind: Literal["sglang", "python", "external"]
    factory: str
    extension_schema_id: str | None = None


class ConfigPatch(BaseModel):
    op: Literal["set", "declare_stage"]
    path: ConfigPath
    value: object | None = None
    declaration: StageDeclaration | None = None
    source: ConfigSource
    layer: int
```

初版 patch protocol 的操作只包含 `set` 和 `declare_stage`：

- `set` 设置一个 typed value；`null` 始终是普通值，用于清空允许为 null 的 optional field；
- `declare_stage` 只由 `document_mode: full` 的 YAML adapter 产生，并携带 typed declaration payload。`name`、`kind` 和 `factory` 必填；stage 使用模型 extension 时提供 `extension_schema_id`。`kind` 选择合法 runtime namespace 和 capability rule，不能先创建 untyped stage 再附加 backend field。

full document 从空 topology 开始，可以声明 stage；partial document 和 CLI `--set` 只能修改已经存在的 stage，不能创建或删除 stage。`stage_order` 中的名称必须与最终声明的 stages 一一对应，不能缺失、重复或引用未知 stage。初版 patch protocol 不提供删除继承 stage/map entry 的操作；需要改变 stage membership 时使用 full document。

初版 patch protocol 始终整体替换 list，不增加 append/remove item 等顺序敏感操作。复杂 list 应优先写 YAML。

`ConfigPath` 不是任意字符串遍历器，而是由 public schema 编译的 path：

- 支持 name-keyed stage；
- 验证字段是否 public/configurable；
- 返回目标 Pydantic type；
- 禁止访问 `factory_args`、derived plans 和 private fields；
- error 中列出相邻合法 path。

## 7.2 `ConfigPatchSet`

adapter 输出有序 patch set。layer 固定为：

| Layer | 来源 |
|---|---|
| 0 | framework/schema defaults |
| 10 | model Python defaults |
| 20 | selected profile/variant |
| 30 | user YAML |
| 40 | Router structured worker patch |
| 50 | worker CLI `--set` 或迁移期 alias |

layer 只解决跨来源 precedence，不解决同层歧义：

- 同一个 document/CLI 中，相同 specificity 重复出现同一 normalized path：报错；
- 同 layer、相同 specificity 的多个来源写同 path：报错，要求调用者明确顺序或合并 document；
- 高 layer 覆盖低 layer：允许并记录 provenance；
- 同一 CLI source 中的 compatibility alias 按 broadcast < role < 显式具体 `--set` 处理；更具体的值覆盖较不具体的值并保留 provenance，相同 specificity 的重复仍报错；
- 同一 Router source 中，`worker_defaults` < 具体 per-worker entry；entry 覆盖 defaults 时保留被遮蔽值的 provenance，相同 scope 内的重复仍报错；
- 同 layer 的 parent subtree 与 child leaf assignment 重叠：报错；
- 跨 layer 时先应用低层：高层 child 可以覆盖低层 parent 的对应 leaf，高层 parent 则整体替换低层 subtree；
- list 始终视为 leaf，不执行按 index merge；
- 通用 `--set` 只能定位 configurable leaf；mapping/subtree replacement 只能使用 full YAML。

YAML parser 必须开启 duplicate-key rejection，并保留每个 mapping/scalar 的文件、行、列信息。重复 YAML key、相同 specificity 下规范化后重复的 path，以及相同 specificity 的 alias/canonical 重复必须在生成 `ConfigPatchSet` 时报告，不能先读成普通 dict 再让后值静默覆盖前值。

## 7.3 `ConfigResolver`

resolver 顺序固定：

1. 收集 adapter patches；
2. 将 V1 path/alias 归一化为 canonical path；
3. 把 selector broadcast 展开为具体 stage leaf patch，并记录 expansion provenance；
4. 结合 layer 执行 specificity、duplicate 和 subtree overlap 检查；
5. 根据 target schema 解析 value；
6. 按 layer 应用到 immutable base tree；
7. 构造完整 `ResolvedPipelineConfig`；
8. 执行 cross-field 和 model capability validation；
9. 冻结配置并输出 provenance map。

resolver 不执行 GPU topology probe、不 import stage factory、不构造 `ServerArgs`。这些属于后续 planner/worker，但只能读取已经 resolve 的 typed fields。

## 7.4 `ResolvedPipelineConfig`

resolved config 与用户 document 分开：

| 对象 | 内容 | 是否允许用户设置 |
|---|---|---|
| `PipelineConfigDocumentV2` | partial/full public document | 是 |
| `ResolvedPipelineConfig` | 所有 defaults 和 patches 应用后的完整 typed config | 通过 resolver 生成 |
| `StagePlacementPlan` | 实际 GPU placement 与 memory accounting | 否 |
| `ProcessTopologyPlan` | OS process membership 和 TP ranks | 否 |
| `DerivedRuntimePlan` | 主进程 topology/hardware probe 可确定的 effective runtime values | 否 |
| `StageWorkerProcessSpec` | spawn payload、endpoint、queues、factory defaults | 否 |
| `WorkerRuntimeDerivation` | 只能在 worker 初始化时确定的 effective values 及 provenance | 否 |
| `ServerArgs` / `ModelConfig` | worker 内 SGLang runtime object | 否 |

`ResolvedPipelineConfig` 使用 frozen model，runtime planner 不得反向修改。resolver 只冻结用户意图，例如 `custom_all_reduce: auto|on|off`，不得依赖 GPU topology 或 hardware probe。

所有可在主进程确定的 topology-dependent effective value 由 runtime planner 写入完整的 `DerivedRuntimePlan`，并记录 topology/hardware source；worker spec 携带当前 worker 需要的 plan slice，worker builder 只能消费，不能补写或反向修改该 plan。

只能在模型加载或 worker 初始化后确定的值进入独立的 `WorkerRuntimeDerivation`，例如依赖实际 loaded backend 的兼容调整。它通过 worker startup diagnostic 和带 source 的 `ServerArgs.override()` 记录 provenance，不反向补写 `DerivedRuntimePlan` 或 `ResolvedPipelineConfig`。

`config explain` 解释用户配置和输入 provenance；runtime planner diagnostic 展示 `DerivedRuntimePlan`，worker startup diagnostic 展示 `WorkerRuntimeDerivation`，共同解释最终 effective value。

## 7.5 canonical runtime namespace

公共 runtime 参数进入 typed namespace：

```text
pipeline.stages.<stage>.runtime.max_seq_len
pipeline.stages.<stage>.runtime.video_fps
pipeline.stages.<stage>.runtime.sglang.mem_fraction_static
pipeline.stages.<stage>.runtime.sglang.max_running_requests
pipeline.stages.<stage>.runtime.sglang.max_total_tokens
pipeline.stages.<stage>.runtime.sglang.encoder_mem_reserve
pipeline.stages.<stage>.runtime.sglang.cuda_graph.*
pipeline.stages.<stage>.runtime.sglang.upstream_args.<field>
pipeline.stages.<stage>.runtime.compile.*
pipeline.stages.<stage>.runtime.decode.*
pipeline.stages.<stage>.runtime.prefill_coalesce.*
pipeline.env_defaults.<NAME>
pipeline.stages.<stage>.env.<NAME>
```

两个环境 path 只保存已配置默认值。解析 configured default 时 stage value 覆盖 Pipeline default；spawn 时派生 `ResolvedProcessEnv`，继承进程环境仍优先。它们不会把任意 `os.environ` 读取收进配置 precedence。

以下字段不再作为 public user interface：

```text
factory_args.server_args_overrides
runtime_overrides
runtime_arg_map
gpu_id
process_total_gpu_memory_fraction
nccl_port
具体 socket/IPC endpoints
```

`pipeline.endpoints.base_path` 仍是用户可配置的分配策略输入。planner 在其下派生具体 socket path/endpoint；只有这些具体派生值禁止用户输入。

`runtime.sglang.upstream_args` 是 stage-local、经过校验的逃生口，用于尚未提升到 Omni typed schema 的 SGLang 字段。其 schema 从精确 pin 的 upstream `ServerArgs` dataclass/schema 生成，同时 pin upstream version 和确定性的 schema fingerprint。它拒绝未知字段，并通过维护的 denylist 拒绝 Omni-owned、derived 或 unsafe 字段，包括 device/process/port/endpoint ownership。已经暴露为 Omni typed field 的字段也禁止出现在 `upstream_args`，从而避免双路径。每个接受的 leaf 都保留 provenance，完整组装后仍执行最终 upstream `ServerArgs` validation。因此该逃生口保持 **一个语义、一个路径**，而不是新增未经校验的 override dict。`kind`/capability 非 SGLang backend 的 stage 完全拒绝 `runtime.sglang`。

`encoder_mem_reserve` 是 canonical SGLang-stage policy field，不再是 factory argument。只有 effective thinker `mem_fraction_static` 保持 `auto`/unset，且没有被任何 user source（包括 YAML、Router、selector expansion 或 alias）通过该 typed canonical path pin 时才合法。`mem_fraction_static` 已是 Omni typed field，因此本来就禁止出现在 `upstream_args`。resolver 根据 provenance 而不只是最终数值验证该约束。runtime derivation 随后从自动 thinker budget 中扣除 reserve 并应用现有 safety floor；它不修改 frozen policy，也不会静默覆盖 pinned fraction。

模型专属、尚未进入公共 schema 的参数放在：

```yaml
pipeline:
  stages:
    talker_ar:
      extensions:
        qwen3_omni:
          partial_start: true
```

extension model 仍必须由模型包注册 Pydantic schema；不允许裸 `dict[str, Any]` 覆盖公共字段。成熟的跨模型功能再从 extension 提升到公共 runtime namespace。

full export 必须物化每个 effective typed/extension value，包括当前隐藏在 `factory_args` 或 Python factory default 中、会定义行为的值。例如 Qwen 模型配置中的 effective `talker_max_seq_len` 是 `32768`，而 factory 签名默认值是 `4096`；导出的 V2 document 必须显式包含 `32768`。在所有这类行为定义 factory default 都进入 public 或注册 extension schema、且 export/import 不依赖隐藏 Python default 前，该模型不能 opt in V2。

## 7.6 role 与 stage 的关系

现有 CLI 通过多个 class method 分别维护 thinker、talker、generation 和 memory role map。V2 在 model config 中统一声明：

```yaml
pipeline:
  roles:
    thinker: thinker
    talker: talker_ar
    generation: talker_ar
    code2wav: code2wav
```

role 只用于：

- 迁移旧 typed CLI alias；
- 模型能力校验；
- 生成面向用户的 shortcut/help。

canonical storage 始终是具体 stage path。resolver 在产生 patch 时解析 role，不在 runtime helper 中重复解析。

## 7.7 开发者新增配置的唯一流程

开发者先根据 owner 判断字段归属：

| 新配置类型 | 应增加的位置 | 不应增加的位置 |
|---|---|---|
| 多模型共享的 Pipeline/runtime 字段 | public typed schema | 手写 CLI helper、裸 `factory_args` |
| 单一模型专属字段 | 模型注册的 extension schema | 全局 `serve()` 签名 |
| HTTP server/process 字段 | `ServerLaunchConfig` | `PipelineConfig` |
| topology/hardware 派生值 | `DerivedRuntimePlan` | 用户 YAML/CLI |
| worker 初始化后才能确定的值 | `WorkerRuntimeDerivation` | resolver 或 Pipeline mutation |
| request-time 字段 | request protocol/schema | 启动期配置 |

新增一个 public typed field 的标准步骤固定为：

1. 在 canonical schema 定义 path、类型、默认值、description 和 capability constraint；
2. 在 resolved config → runtime consumer adapter 中消费该 typed field；
3. 从 schema metadata 自动生成 JSON Schema、CLI path help 和 YAML/CLI 等价性测试；
4. 如需兼容旧入口，只在 `compat.py` 添加限时 old path → canonical path 映射，并声明删除 Phase；
5. 不在 `serve.py` 新增参数专属 mutation helper。

Code review 可以据此拒绝任何绕过 canonical schema 的新公共配置入口。

# 8. YAML 与 CLI 的职责边界

新接口明确以下规则：

| 配置类别 | YAML | CLI | 原因 |
|---|---|---|---|
| 模型、stage topology | canonical source | 只通过 `--config`/`--model-path` 选择 | 结构复杂，需要持久化 |
| GPU/TP/process placement | canonical source | 可用 `--set` 临时覆盖 | 属于 Pipeline，需统一 validation |
| SGLang/runtime 参数 | canonical source | 可用 `--set` 临时覆盖 | 同一 path、同一类型、同一校验 |
| HTTP bind/log/API policy | 不进入 Pipeline YAML | 专用 CLI | 属于服务进程，不属于模型 Pipeline |
| Endpoint 分配策略（`endpoints.base_path`） | canonical source | 可用 `--set` 临时覆盖 | 用户可配置的分配根目录 |
| 具体 runtime-derived endpoint/port/gpu id | 不可设置 | 不可设置 | planner/runner owner |
| Router worker pool | Router YAML | Router CLI 只选择 source/路由参数 | 与 worker Pipeline 分层 |

CLI 和 YAML 因而仍有两种语法，但只有一套 Pipeline 语义：

```text
YAML leaf
→ ConfigPatch(path, value, file source)

CLI --set
→ ConfigPatch(path, value, cli source)
```

# 9. Router 接口重构

## 9.1 structured worker config

launcher YAML 移除自由字符串 `worker_extra_args`，改用共享默认值和显式 per-worker entry：

```yaml
schema_version: 2

launcher:
  backend: local
  num_workers: 4
  worker_host: 127.0.0.1
  worker_base_port: 8011
  wait_timeout: 600

  worker_defaults:
    config: examples/configs/qwen3_omni_colocated_h20_v2.yaml
    service:
      model_name: qwen3-omni
    capabilities: [speech]
    patches:
      pipeline.stages.thinker.runtime.sglang.max_running_requests: 32

  workers:
    - index: 0
      gpus: ["0"]
      capabilities: [speech, streaming]
      patches:
        pipeline.stages.thinker.runtime.sglang.max_running_requests: 48
    - index: 1
      gpus: ["0"]  # 与 worker 0 有意共享 GPU。
    - range: "2-3"
      gpus: ["GPU-3b6e...", "MIG-GPU-7d2a.../1/0"]
      capabilities: [chat]
```

`worker_defaults` 提供共享的 `config` 或 `model`（二选一）、service value、capability 和 patch。每个 `workers` entry 用 `index` 选择一个 worker，或用 `range` 选择包含端点的一组 worker，并可覆盖 `gpus`、capability 和 patch。同一 Router source 内先展开 `worker_defaults`，再应用具体 entry；entry patch 覆盖同 path default 并保留 provenance，不按同 scope duplicate 报错。entry 自身同 path 重复仍报错。允许不同 worker 使用重复的 physical GPU set：示例明确让 worker 0 和 1 都使用 physical GPU `0`。physical identifier 按原样保留，包括形似整数的字符串、UUID 和 MIG identifier；Router 不把它们转换成 logical ID。entry 必须恰好覆盖每个 worker，重叠的 index/range selector 报错。

`num_gpus_per_worker` 只在没有任何显式 worker `gpus` 时作为 fallback；此时 Router 自动、互斥地切分可用设备。它不能与显式 per-worker GPU set 同时使用。分配实际可见设备集合后，Router 按 `0..len(worker.gpus)-1` 校验 Pipeline logical GPU ID，而不是使用全局数量。

## 9.2 worker 传递方式

初版结构化 Router 实现可以继续生成 argv，但只生成机器接口：

```text
sgl-omni serve
  --config ...
  --config-patch-file /tmp/.../worker-0-patches.json
```

patch file 内容是序列化的 `ConfigPatchSet`，不是 shell fragment。worker 验证 source 和 schema version 后交给同一个 resolver。

后续如果 Router 与 worker 改为 Python API 或 supervisor protocol，仍复用 `ConfigPatchSet`，不改变配置语义。GPU visibility 继续由 Router 通过 subprocess env 管理，因为它是进程资源边界，不属于 Pipeline patch。

Router 和 worker 使用同版本的 schema registry 与 `ConfigResolver`。Router 在创建 subprocess 前，对每个已选择 worker 的 `defaults + entry + config/model + patches` 执行完整 dry-run resolve：

- 相对配置路径以 launcher YAML 所在目录为基准；
- Router 必须先解析已选择的 model/config identity，并精确加载该 worker 引用的 extension schema；无法完成此 lookup 时启动失败；
- path、type、role 或 extension 无法解析时，在启动任何 worker 前失败；
- worker 收到 patch file 后再次校验 schema version 和内容，但不得采用不同的 merge 规则。

GPU 使用两个明确坐标系：

- Router 是唯一有权分配 physical GPU 的组件，通过 `CUDA_VISIBLE_DEVICES` 建立每个 worker 的可见设备集合；显式 assignment 可以有意跨 worker 重叠；
- Pipeline `placement.gpus` 始终使用 worker-local logical GPU ID，即可见设备重新编号后的 `0..N-1`。

Router dry-run 必须验证所有 logical GPU ID 均位于 `[0, len(actual worker visible GPU set))`，并验证 TP size、GPU list 和 worker allocation 一致。Pipeline 不允许直接引用宿主机 physical GPU ID。这个前置条件不要求 Router import 或加载所有已注册模型，只要求加载已选择的 worker model 及其引用的 extension。

## 9.3 Router 与 worker 的优先级

managed worker 的 layer 固定：

```text
模型默认 < worker Pipeline YAML < Router worker.patches < worker CLI emergency patch
```

默认不允许 `worker CLI emergency patch`，只有调试模式显式开启。生产配置由 Router YAML 完整决定，避免 supervisor 生成值和手工 argv 再次竞争。

# 10. 配置冲突与错误模型

## 10.1 冲突分类

| 冲突类型 | 示例 | V2 行为 |
|---|---|---|
| 同 source 重复 | 两个 `--set` 写同 path | 启动前报错 |
| alias specificity | broadcast alias 与 role alias 写同 leaf | 更具体的 role value 生效，并记录被遮蔽值 |
| 相同 specificity 的 alias/canonical 重复 | 两个 role alias 或两个显式 `--set` 写同 leaf | 启动前报错 |
| typed/extension 重叠 | extension 试图设置公共 SGLang 字段 | schema 注册失败 |
| parent/child 重叠 | 两个同层 patch source 分别设置 `runtime.sglang` 及其 child | 同层报错；跨层允许并记录 |
| owner 违规 | 用户设置 `gpu_id` 或具体 endpoint | path 不公开，解析失败；`endpoints.base_path` 仍可配置 |
| capability 不支持 | 非 SGLang stage 设置 `runtime.sglang` | model validation 失败 |
| cross-field 不一致 | TP=2 但只有一个 GPU | resolved model validation 失败 |
| deprecated document/canonical 重复 | 同一 document layer 中 V1 `runtime_overrides` 与 V2 field | migration error；CLI alias 改按显式 specificity 规则处理 |

## 10.2 原子性

resolver 在内存中的 immutable tree 上应用所有 patches。任何 path、type、conflict 或 validation 失败时，不返回 partial config，也不进入 runtime planner。

错误必须包含：

- canonical path；
- source location；
- 原值及其 source；
- 冲突规则；
- 可执行的修复建议。

例如：

```text
duplicate configuration for
pipeline.stages.thinker.runtime.sglang.mem_fraction_static

cli:--thinker-mem-fraction-static = 0.70
cli:--generation-mem-fraction-static = 0.65

The two role aliases have equal specificity and resolve to the same stage.
Remove one alias or use an explicit concrete `--set`.
```

# 11. 配置编译与运行时边界

V2 编译链为：

```text
① Parse input documents
→ ② Normalize patches
→ ③ Resolve + validate + freeze
→ ④ Build placement/process plans
→ ⑤ Build derived runtime plan
→ ⑥ Build worker specs
→ ⑦ Construct worker runtime objects
```

只按 **① → ② → ③ → ④ → ⑤ → ⑥ → ⑦** 阅读：

- ②之后所有输入都使用 canonical path；
- ③是最后一次修改用户配置的步骤；
- ④只生成派生 plan，不回写 Pipeline；
- ⑤保存主进程 topology/hardware probe 可决定的 effective runtime values；
- ⑥在父进程解析与 factory 签名无关的参数，并整理进跨进程 DTO；
- ⑦ import factory，只注入缺失的签名相关默认值，再构造 `ServerArgs` 和 `ModelConfig`；只能在该阶段确定的调整记录到 `WorkerRuntimeDerivation`。

当前 `_apply_tensor_parallel_server_args_overrides()` 会基于 topology probe 修改 stage server args。V2 将其拆为：

1. 用户配置表达 `custom_all_reduce: auto|on|off`；
2. placement planner 产生 topology capability；
3. 主进程 runtime planner 根据两者产生 `DerivedRuntimePlan`；
4. worker builder 只消费 plan 中的 effective `ServerArgs` value；
5. 不修改 frozen Pipeline，也不在 worker 重新执行同一 topology 决策。

其他 topology-dependent 参数遵循同样模式。

# 12. 改动范围

## 12.1 配置 schema 与 resolver

| 模块 | 改动 |
|---|---|
| `sglang_omni/config/schema.py` | 增加 V2 public schema、name-keyed stages、typed runtime namespaces、frozen resolved model |
| `sglang_omni/config/patch.py`（新增） | `ConfigPath`、`ConfigSource`、`ConfigPatch`、duplicate/overlap 检查 |
| `sglang_omni/config/resolver.py`（新增） | selector expansion、specificity/layer merge、type conversion、provenance、完整 validation |
| `sglang_omni/config/compat.py`（新增） | V1 YAML/dotted/typed CLI 到 V2 patch 的限时 adapter |
| `sglang_omni/config/sglang_schema.py`（新增） | 生成并 fingerprint 精确 pinned `ServerArgs` passthrough schema，执行 ownership/unsafe denylist |
| `sglang_omni/models/registry.py` | 注册 V2 model defaults、roles、stage kind/capability 和 extension schemas |

`ConfigManager` 的文件加载职责并入 adapter/resolver。旧类在迁移期保留 facade，最终删除自定义 dotted merge。

## 12.2 CLI

| 模块 | 改动 |
|---|---|
| `sglang_omni/cli/serve.py` | 参数解析后只构造 `ServerLaunchConfig` 和 patches；删除业务 mutation helper |
| `sglang_omni/cli/config.py` | 增加 schema/validate/resolve/explain/migrate/export、selector expansion 展示和环境诊断 |
| `sglang_omni/cli/__init__.py` | 关闭未知 option；开放覆盖统一走显式 `--set` |

迁移期 typed CLI alias 可以继续在 `serve()` 签名中出现，但实现由 alias registry 自动生成 patch。

## 12.3 runtime

| 模块 | 改动 |
|---|---|
| `sglang_omni/config/runtime.py` | 把 typed resolved runtime 转成父进程已解析的静态 factory kwargs；不再解决用户来源冲突 |
| `sglang_omni/pipeline/runtime_config.py` | 只读 frozen resolved config，输出 placement/process 和 `DerivedRuntimePlan` |
| `sglang_omni/pipeline/mp_runner.py` | 从 resolved config/plans 构造 spec 和 `ResolvedProcessEnv`，保持 static args/signature injection 边界 |
| `sglang_omni/pipeline/stage_workers.py` | 应用已配置环境默认值但不覆盖继承值，并输出 process environment 诊断 |
| `sglang_omni/scheduling/sglang_backend/server_args_builder.py` | 从 typed SGLang config 构造 `ServerArgs`，保留明确的 derived runtime overrides |

模型 factory 可以暂时继续接收 `server_args_overrides`，但该 dict 只能由 typed runtime adapter 内部生成，不再是 public YAML/CLI 入口。

## 12.4 Router

| 模块 | 改动 |
|---|---|
| `sglang_omni_router/launcher/config.py` | `worker_defaults`、index/range 选择的 `workers`、capability、physical GPU identifier 和 patch schema |
| `sglang_omni_router/launcher/local.py` | 生成 patch file，不拼接 `worker_extra_args` |
| `sglang_omni_router/serve.py` | worker source validation 与 structured launcher 接入 |

# 13. 分阶段兼容方案

V2 采用分阶段迁移，不要求一个版本内修改所有配置，但每个 deprecated 入口都有删除阶段。

## 13.1 Phase 0：建立 resolver，不改变默认行为

- 首先运行 `2b45073c` 上完全未改动的 latest-main V1 实现，冻结独立 oracle：记录最终 Pipeline shape、factory kwargs、placement/process plans、worker specs 和用户可控 `ServerArgs` 字段。在任何 V1 输入经过新 adapter/resolver 前先生成这些 legacy golden；
- 在当前行为外增加只读 provenance 重建和 `config resolve/explain`，此时不让新 resolver 成为 authoritative path；
- 引入 `ConfigPatch`、provenance 和 `config resolve/explain`；
- 在不替换 V1 执行路径的 shadow mode 中，当前模型默认、V1 YAML、dotted CLI 和 typed CLI 同时通过 compatibility adapter 产生 patch；
- 将 adapter/resolver 输出与独立的 untouched-V1 oracle 比较；V1 adapter 与 V2 通过同一 resolver 的对比只是补充，不能替代独立 oracle；
- runtime 仍消费现有 `PipelineConfig` shape；
- 新 resolver 发现的 V1 重复来源先记录 diagnostic，不改变当前成功/失败结果；
- 内部测试的 V2 输入从一开始执行严格 duplicate/overlap 规则。

该阶段目标是先得到唯一 merge engine，而不是立即改变用户语法或 validation 行为。Phase 1 起，V2 输入严格报错，V1 输入对新发现的重复来源 warning；Phase 2 起，V1 同义重复也报错。

## 13.2 Phase 1：V2 opt-in

- 支持 `schema_version: 2`；
- 新文档和 export 默认展示 V2，可通过 flag 保留 V1 export；
- Router 支持 structured worker 配置，同时兼容 `worker_extra_args`；
- typed CLI alias 打印 canonical `--set` 替代方式；
- telemetry/log 统计 deprecated path，不上传具体配置值。

V2 document 不允许出现 V1 `stage_overrides`、`runtime_overrides` 或 public `factory_args.server_args_overrides`。
V1 输入继续保留当前最终值语义，但新发现的重复来源输出 warning。
模型只有在所有隐藏于 `factory_args` 或 factory default 中、会定义行为的值都进入 public 或注册 extension schema，并能通过 full export round-trip 后，才允许 opt in V2。

## 13.3 Phase 2：V2 默认

- `config export`、官方 examples、cookbook 和 Router examples 全部切换到 V2；
- 没有 `schema_version` 的 YAML 按 V1 处理并输出 warning；
- dotted unknown option 关闭，要求显式 `--set`；
- runtime typed CLI alias 输出 deprecation warning；
- V1 deprecated/canonical 同义重复改为启动错误；
- `worker_extra_args` warning，并提供 `config migrate` 自动转换可识别参数。

## 13.4 Phase 3：删除旧入口

在下一个明确公布的 major/minor compatibility boundary：

- 删除 `stage_overrides`；
- 删除 public `runtime_overrides`；
- 禁止 YAML 设置 public `factory_args.server_args_overrides`；
- 删除动态 unknown option dotted CLI；
- 删除 runtime 专用 typed CLI alias；
- 删除 Router `worker_extra_args`；
- 删除 `ConfigManager.merge_config()` 的 V1 path traversal。

读取 V1 YAML 时给出明确错误和离线 migration 命令，不在 server 内继续隐式兼容。

## 13.5 兼容行为表

| 当前接口 | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| V1 Pipeline YAML | 支持 | 支持 + 可迁移 | warning | 拒绝 |
| `stage_overrides` | 支持 | warning | warning | 拒绝 |
| `runtime_overrides` | 支持 | warning | warning | 拒绝 |
| dotted unknown CLI | 支持 | warning | 拒绝并提示 `--set` | 删除 |
| runtime typed CLI | 支持 | warning | warning | 删除 |
| V2 YAML | 内部测试 | opt-in | 默认 | 唯一格式 |
| Router `worker_extra_args` | 支持 | warning | warning | 拒绝 |
| structured Router worker | 内部测试 | opt-in | 默认 | 唯一格式 |

# 14. 校验与测试

校验分为 resolver 和 runtime planning 两层。前者证明“用户声明唯一且类型正确”，后者证明“解析后的 topology 可以执行”。

## 14.1 Resolver 单元测试

| 测试类别 | 检查内容 |
|---|---|
| path | 合法 field、unknown field、private/derived field、stage name |
| type | scalar、null、list、dict、enum、invalid coercion |
| document structure | duplicate YAML key、full/partial mode、stage declaration、`stage_order` 完整性 |
| precedence | defaults、profile、YAML、Router patch、CLI、source layer 主导 selector specificity |
| selectors | capability/role/stage/exclude 匹配、展开、空选择、具体 stage storage/export |
| duplicates | 同 source/specificity、同 layer、alias specificity、parent/child |
| cross-layer subtree | 高层 child 覆盖低层 parent、高层 parent 替换低层 subtree、list 整体替换 |
| provenance | 每次覆盖的旧值、source、location、deprecated 标记 |
| atomicity | 任意 patch 失败时不产生 partial resolved config |
| frozen config | resolver 返回后禁止 assignment |
| role | role 到 stage 的唯一解析、unknown/unsupported role |
| extension | 注册 schema、未知 key、禁止覆盖公共 namespace、隐藏 default 的 full-export round trip |
| upstream args | 精确 pinned `ServerArgs` version/fingerprint、未知字段、denylist、typed field 重复、per-leaf provenance、最终 `ServerArgs` validation |
| environment | `env_defaults`/stage `env` merge、继承环境不被覆盖、派生 `ResolvedProcessEnv`、外部读取诊断 |

## 14.2 CLI/YAML 等价性

对每个 public typed field建立参数化测试：

```text
YAML canonical leaf
==
CLI --set canonical path
==
迁移期 typed CLI alias
```

三种输入必须产生相同 `ResolvedPipelineConfig`，只有 provenance 不同。参数表由 schema metadata 自动生成，避免手写测试遗漏新字段。

## 14.3 V1 migration golden

选取官方 examples：

- Qwen3-Omni text/speech/colocated；
- Qwen3-TTS；
- Higgs、MOSS、FishAudio；
- ASR 单 stage；
- Ming 完整 stages YAML；
- TP 和 process isolation 配置。

每个 case 首先从完全未改动的 latest-main V1 实现捕获独立 golden，再把 V1 adapter 和迁移后的 V2 document 分别与该 oracle 比较：

- stage topology 和顺序；
- placement/TP/process；
- typed runtime；
- factory 最终 kwargs；
- placement/process plans；
- worker specs；
- `ServerArgs` 用户可控字段。

runtime-derived endpoint 和临时 port 使用结构比较，不比较随机值。

## 14.4 冲突回归

必须覆盖：

- typed `mem_fraction_static` 与旧 `server_args_overrides`；
- global 与 per-role CLI；
- broadcast alias、role alias 和显式 `--set` specificity，包括 shadowed provenance；
- selector broadcast 与具体 stage override；
- `--encoder-mem-reserve` 与 explicit fraction；
- `tp_size` 与 GPU list；
- generation role 与 thinker role 指向同/不同 stage；
- deprecated path 与 canonical path；
- Router patch 与 worker CLI；
- extension 与 public namespace。

每个冲突断言 canonical path 和两个 source 都出现在错误中。

## 14.5 Runtime 边界

验证：

- runtime planner 不修改 frozen config；
- topology-dependent effective value 只进入 `DerivedRuntimePlan`；
- worker builder 只消费 `DerivedRuntimePlan`，不能补写或重复决策；
- worker-only effective value 只进入 `WorkerRuntimeDerivation` 并保留 source；
- child process 不重新 resolve user config；
- factory signature injection 只补缺失的 derived defaults；
- user 不能设置 `gpu_id`、process cumulative budget、NCCL port 和具体 endpoint，但 `endpoints.base_path` 仍可配置；
- TP env mapping 与当前行为一致；
- no-config/default model path 的 worker specs 与 golden 一致；
- `ServerArgs.override()` 仍记录 worker 内 derived mutation source。

## 14.6 Router

验证：

- structured config 生成的 worker command 不包含 shell-like extra args；
- patch file 每 worker 独立、schema version 正确；
- model/config 互斥；
- Router 使用同版本 registry/resolver，在启动 subprocess 前完成 dry-run；
- launcher-relative config path 和 model extension schema 正确解析；
- Router YAML path/type/role/extension 在启动 subprocess 前失败；
- GPU 继续通过 environment 传递；
- physical GPU assignment 与 worker-local logical GPU ID 转换正确；
- 校验 index/range coverage、per-worker capability/patch 和 selected-model/extension lookup；
- 校验 `worker_defaults` < per-worker entry 的覆盖与 provenance，以及各自 scope 内的 duplicate rejection；
- 保留跨 worker 重复的 physical GPU set，包括两个 worker 共享 GPU `0`；
- 形似整数、UUID 和 MIG physical identifier 原样 round-trip；
- logical GPU 根据每个 worker 的实际 visible set 检查越界，TP/allocation 不一致时启动前失败；
- 只有没有显式 `gpus` 时才执行互斥的 `num_gpus_per_worker` 自动切分；
- V1 `worker_extra_args` migration 能识别 `--config`、typed alias 和 `--set`；
- 无法安全迁移的任意 shell token 明确失败，不静默丢弃。

## 14.7 Phase 准入标准

每个迁移 Phase 都必须有 compatibility matrix 测试，断言每种旧入口的：

- success、warning 或 error；
- 最终 resolved value；
- provenance/diagnostic；
- 推荐的 migration command。

进入下一 Phase 前必须满足：

1. 官方 Pipeline 和 Router examples 全部通过 migration golden；
2. structured Router 至少完成一条多 worker E2E；
3. runtime 不再直接读取 public compatibility fields；
4. 当前 Phase 的 warning/error matrix 和完整 compatibility matrix 在 CI 中固定；
5. 官方 examples、文档、测试和 CI 调用全部完成迁移；
6. release notes 已提前公布在预定的 `N` 个 release 后删除；
7. 进入删除 Phase 前，这 `N` 个已公布 release 已实际发布。

repo state 和 release history 是可判定 gate。本地或显式 opt-in telemetry 只能用于诊断剩余使用，不能用不可观测的全局 deprecated usage threshold 阻止或授权 Phase 切换。

# 15. 发布与文档

发布 V2 时同步提供：

1. V1 → V2 migration guide；
2. CLI/YAML 职责表；
3. canonical path reference 和生成的 JSON Schema；
4. `config resolve/explain` 调试示例；
5. deprecated 入口删除版本；
6. 官方 examples 的 V1/V2 diff；
7. Router structured worker 示例。

CLI `--help` 不再列出全部模型 runtime 字段，只说明 `--set` 并指向：

```bash
sgl-omni config schema
sgl-omni config resolve
```

模型特定字段由选定 model config/extension schema 动态展示，而不是把所有模型 option 固定到一个全局 `serve()` 签名。

# 16. 预期结果

重构完成后的稳定关系是：

```text
YAML / CLI / Router
        ↓
同一种 ConfigPatch
        ↓
唯一 ConfigResolver
        ↓
冻结的 ResolvedPipelineConfig
        ↓
只读 runtime planners
```

YAML 与 CLI 保留不同使用场景：

- YAML 保存完整、可 review、可复现的 Pipeline；
- CLI 选择配置、设置服务进程参数，并用 `--set` 做临时 Pipeline patch；
- Router YAML 结构化引用 worker config 和 patches。

它们不再分别定义 runtime 参数的类型、默认值、优先级和 mutation 逻辑。新增一个公共配置字段时，只需要在 canonical schema 定义一次；CLI、JSON Schema、文档、patch type 和等价性测试都从该定义生成。
