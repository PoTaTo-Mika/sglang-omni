# Configuration Interface Refactoring Design Proposal

# 1. Background and Scope

SGLang-Omni currently supports several configuration methods: model Python defaults, Pipeline YAML, dynamic dotted CLI options, dedicated typed CLI options, environment variables, and Router launcher YAML. Having multiple entry points is not itself a problem: YAML is suitable for storing static parameters for a complete deployment, CLI options are suitable for selecting configurations and applying dynamic temporary overrides, and environment variables are suitable for process-level injection. However, the current configuration logic has three main problems:

1. **Inconsistent processing logic**: When the same configuration item is supplied through different entry points, it does not follow the same processing code. For example, when both YAML and CLI can set the same parameter, field addressing, type conversion, and parameter validation may need to be implemented repeatedly.
2. **Unclear extension path for developers**: When adding a configuration item, developers do not know whether it should go into the typed schema, a CLI option, `factory_args`, `runtime_overrides`, or model-specific configuration, nor which modules must be updated together.
3. **Unclear modification path for users**: Users do not know at which YAML level a configuration should be written, which CLI flag to use, or which value ultimately takes effect when multiple entry points set it simultaneously.

In summary, the refactoring must address all three questions—whether the implementation is unified, how developers extend it, and how users configure it—rather than merely replacing the current merge implementation.

## 1.1 One-Sentence Proposal

**Define each parameter only once; use YAML for long-term storage and CLI for temporary overrides; regardless of where a value comes from, pass it to the same code for parsing and validation.**

Before the refactoring, the same parameter may have multiple processing paths:

```text
The same parameter
├── YAML typed field
├── stage_overrides
├── runtime_overrides
├── factory_args
├── dotted CLI
└── typed CLI helper
```

Each path may contain its own field lookup, type conversion, precedence, and validation code.

After the refactoring, a parameter has only one official address:

```text
One parameter
└── One configuration path
    ├── YAML: long-term storage
    ├── CLI --set: temporary override
    └── Router: structured input
            ↓
       The same configuration processor
            ↓
       Final read-only configuration
```

For example, `max_running_requests` is defined only as:

```text
pipeline.stages.thinker.runtime.sglang.max_running_requests
```

YAML can persistently set it:

```yaml
pipeline:
  stages:
    thinker:
      runtime:
        sglang:
          max_running_requests: 32
```

CLI can temporarily override the same path:

```bash
sgl-omni serve \
  --config config.yaml \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=32
```

Both forms ultimately enter the same type conversion, range validation, conflict detection, and precedence handling code.

## 1.2 Solutions Corresponding to the Three Goals

| Current problem | Rule after refactoring |
|---|---|
| YAML and CLI duplicate parameter processing | Convert every entry point into a unified configuration modification first, then pass it to the same processor |
| Developers do not know where to add a parameter | Add shared parameters only to the public schema; model parameters only to model extensions; service parameters only to server config |
| Users do not know where to modify a value | Put long-term configuration in YAML, use `--set` for temporary overrides, and use dedicated CLI options for service parameters such as host/port/log |

When adding a shared parameter, a developer only needs to define its type, default, and validation once in the schema, then consume it in the runtime adapter. CLI help, YAML Schema, documentation, and equivalence tests are all generated from this definition.

Users do not need to understand `factory_args`, `runtime_overrides`, or the internal planner. They only need to find the parameter's official path and decide whether this change should be "stored long-term in YAML" or "temporarily overridden through CLI."

## 1.3 Terminology in This Document

The remainder of this document uses three technical names for the simple flow above:

- **canonical path**: the unique, official configuration address of a parameter;
- **ConfigPatch**: a modification record that "sets an address to a value," while also recording whether the source is YAML, CLI, or Router;
- **ConfigResolver**: the configuration processor that uniformly performs type conversion, precedence handling, conflict detection, and validation.

These three names are not three new interfaces, but three stages in the same configuration processing pipeline.

This document discusses only startup-time configuration. It does not change the request protocol, Stage payloads, scheduling algorithms, or field definitions in SGLang's internal `ServerArgs`. The code revision checked for the "current implementation" in this document is `2b45073c`; see `docs/developer_reference/configuration.md` for the complete current state.

Tracking issue: [#1466](https://github.com/sgl-project/sglang-omni/issues/1466)

Chinese version: [配置接口重构设计 Proposal](./configuration_interface_refactor_proposal.md)

The refactoring goals are:

1. **Unify configuration processing code**: Each user-facing semantic has only one canonical config path; YAML, CLI, and Router all enter the same parsing, type conversion, precedence, validation, and conflict-handling flow.
2. **Establish a single developer extension interface**: Adding a shared configuration item modifies the typed schema only once; CLI, JSON Schema, documentation, patch types, and equivalence tests are generated from the schema. Model-specific parameters use an explicit extension schema rather than arbitrarily choosing among multiple pass-through dicts.
3. **Establish a discoverable and explainable user interface**: Clearly define the responsibilities of YAML and CLI, and provide schema, validate, resolve, explain, and migrate tools so users can find valid configuration paths and confirm the source of final values.

Configuration should be resolved once, fully validated, and frozen before entering placement/runtime; old interfaces remain compatible through phased migration, but two semantic systems will not be retained permanently.

This document does not require all model configuration to be rewritten at once. The new interface first stabilizes the configuration boundary, then gradually migrates model-specific parameters.

# 2. Current Configuration Interfaces and Execution Flow

## 2.1 Configuration Sources

The current public configuration sources can be divided into seven categories:

| Source | Current entry point | Main implementation |
|---|---|---|
| Model defaults | `PipelineConfig` subclasses, `Variants` | `models/*/config.py`, `config/manager.py` |
| Pipeline YAML | `sgl-omni serve --config` | `ConfigManager.from_file()` |
| Compact YAML overrides | `stage_overrides`, `runtime_overrides` | `config/manager.py`, `config/runtime.py` |
| Dynamic CLI | `--stages.thinker... VALUE` | `ConfigManager.parse_extra_args()` |
| Dedicated CLI | `--mem-fraction-static`, etc. | `apply_*_cli_overrides()` in `cli/serve.py` |
| Configured environment defaults | Pipeline `env_defaults`, per-stage `StageConfig.env` | `config/schema.py`, `pipeline/mp_runner.py`, `pipeline/stage_workers.py` |
| Router launcher | `--launcher-config`, `worker_extra_args` | `sglang_omni_router/launcher/*` |

`sgl-omni serve` enables `ignore_unknown_options` for unknown options, so Typer first parses 41 dedicated parameters, after which the remaining tokens are parsed by `ConfigManager` as dotted overrides. The two types of CLI options are applied at different stages.

Configured environment values have a narrower contract than the configuration layers below. Pipeline `env_defaults` are canonical configured defaults; `StageConfig.env` overlays them for that stage. At spawn, these defaults fill only variables absent from the inherited process environment and therefore never override `os.environ`. The resulting `ResolvedProcessEnv` is a derived process artifact that combines the inherited environment, configured defaults, and runner-owned additions such as rank/device variables; it is not another user precedence layer. Direct `os.environ` reads in libraries or runtime code remain external inputs and must be surfaced by environment diagnostics where relevant, not collected into one generic configuration layer.

## 2.2 Current Merge Order

The current main process constructs the final `PipelineConfig` in the following order:

1. If `--config` is present, construct the configuration class using `config_cls` from YAML; otherwise, identify the model architecture through `model_path`;
2. Deep-merge YAML `stage_overrides` into typed `StageRuntimeConfig`;
3. Apply the dotted CLI options in `ctx.args` to the dict produced by `model_dump()`, then rebuild the Pydantic model;
4. An explicit `--model-path` overrides `model_path` from YAML;
5. `serve.py` runs the memory, TP/GPU, process, CUDA graph, compile, decode, prefill, and generation batch helpers in sequence;
6. `launch_server()` passes HTTP/API parameters separately from `PipelineConfig`;
7. Runtime preparation constructs placement, process topology, endpoints, and worker specs;
8. Before spawn, the parent resolves signature-independent factory arguments (`factory_args`, `runtime_overrides`, and typed runtime mappings) into the worker spec;
9. In the child, the worker imports the factory, injects only missing defaults whose applicability depends on its signature, and then the factory constructs `ServerArgs` and `ModelConfig`.

| Order | Processing stage | Input and output |
|---|---|---|
| 1 | Select base configuration | Model Python defaults or Pipeline YAML → `ConfigManager` |
| 2 | Apply compact YAML overrides | `stage_overrides` → typed runtime |
| 3 | Apply dynamic CLI | dotted CLI → rebuilt `PipelineConfig` |
| 4 | Apply dedicated CLI | typed CLI helpers → in-place modification of `PipelineConfig` |
| 5 | Runtime planning | `PipelineConfig` → placement, process topology, parent-resolved static factory args and worker spec |
| 6 | Worker construction | worker spec → import factory → inject missing signature-dependent defaults → `ServerArgs` |

This order appears to provide precedence, but precedence is not a single unified rule. Different helpers choose to "override," "reject," "synchronize two aliases," "write both `factory_args` and `runtime_overrides`," or "wait until the worker to report a conflict."

## 2.3 Current Factory Parameter Path

Stage factory kwargs are combined across an explicit parent/child boundary:

```text
factory_args
< runtime_overrides
< typed runtime mapping
→ parent-resolved static factory args in worker spec
+ child-injected missing signature-dependent defaults
```

Here, `server_args_overrides` performs an additional dict merge between `factory_args` and `runtime_overrides`; typed `runtime.max_seq_len` and `runtime.video_fps` are renamed through `runtime_arg_map`; typed `mem_fraction_static` is ultimately written back to `server_args_overrides`.

The parent performs all merges that do not require inspecting the factory signature before spawn. `model_path`, `gpu_id`, and `total_gpu_memory_fraction` are not ordinary user factory args: the parent computes their candidate values, and after importing the factory the child injects a candidate only when the signature declares that parameter and the resolved static args did not already contain it. `gpu_id`, the cumulative process budget, NCCL port, and concrete IPC/socket endpoints are derived by the planner/runner.

# 3. Problems with the Current Design

## 3.1 Multiple User Entry Points for the Same Semantic

`mem_fraction_static` is the most complete example:

| Entry point | Write location | Validation location |
|---|---|---|
| YAML typed runtime | `stage.runtime.sglang_server_args.mem_fraction_static` | Pydantic schema |
| YAML compatibility override | `runtime_overrides.<stage>.server_args_overrides.mem_fraction_static` | runtime adapter |
| Full stage YAML | `factory_args.server_args_overrides.mem_fraction_static` | runtime adapter |
| dotted CLI | Can write any path above | Pydantic/runtime adapter after merge |
| global typed CLI | `--mem-fraction-static` | `apply_mem_fraction_cli_overrides()` |
| role typed CLI | `--thinker-mem-fraction-static`, etc. | The same helper and role map |

The current code must explicitly detect duplicate sources between typed runtime and `server_args_overrides`. Such detection can only cover known conflicts and is easily omitted when a new source is added.

Generation batch configuration has the same kind of problem. `max_running_requests` can come from:

- Model factory defaults;
- YAML `factory_args`;
- YAML `runtime_overrides`;
- dotted CLI;
- `--thinker-max-running-requests`;
- `--max-running-requests`;
- Derivation or validation by the model's generation batch policy.

Users see one feature, while the code maintains multiple paths.

## 3.2 YAML Serves as Both a "Configuration Instance" and a "Patch"

Current YAML can be:

1. A template selection containing only `config_cls` and `model_path`;
2. A compact patch with `stage_overrides`;
3. A factory patch with `runtime_overrides`;
4. A resolved-like configuration with full `stages`.

The four forms enter different merge logic. In particular:

- `stage_overrides` permits only typed `runtime`;
- `runtime_overrides` is a free-form dict;
- Full `stages` replaces the model's default list as a whole;
- These forms can be mixed in the same YAML.

Consequently, "where should this field be written" cannot be answered solely from the field's semantics; it also depends on which YAML form the user chose.

## 3.3 CLI Is Both an Interface Layer and a Business Logic Layer

`cli/serve.py` not only declares options, but is also responsible for:

- Resolving public roles to model stages;
- Modifying nested Pydantic objects;
- Synchronizing `tp_size` and `parallelism.tp`;
- Writing parameters into `factory_args` and `runtime_overrides`;
- Checking whether a factory belongs to the supported set;
- Performing GPU topology probes;
- Re-running part of Pydantic validation.

As a result, adding a runtime parameter usually requires modifying all of the following:

1. Typer signature;
2. CLI helper;
3. Role map;
4. Runtime adapter or factory;
5. Tests;
6. Documentation.

CLI is no longer an input adapter, but a second configuration compiler.

## 3.4 In-Place Mutation Bypasses Schema Validation

The current Pydantic models do not enable `validate_assignment=True`. `ConfigManager.merge_config()` performs dump, deepcopy, and reconstruction, and therefore can run full validation; most typed CLI helpers directly modify:

- `stage.gpu`
- `stage.tp_size`
- `stage.runtime.*`
- `stage.factory_args`
- `pipeline_config.runtime_overrides`

Helpers must ensure atomicity and complete validation themselves. Some paths manually rebuild the entire config, while others perform only local range checks. Whether a configuration is valid depends on the specific helper that modified it rather than on the config schema itself.

## 3.5 Dynamic Dotted CLI Lacks a Stable Contract

Dynamic CLI converts `-` to `_`, traverses dicts/lists by `.`, and supports only bool, None, int, float, and string scalars. It has the following limitations:

- It can write arbitrary internal fields, including compatibility fields that should not be public;
- The stage list supports both numeric indices and names, and indices become unstable when the exported configuration changes;
- Lists/dicts have no unified encoding;
- A nonexistent path may raise `KeyError`/`IndexError`, with inconsistent error formats;
- Available paths cannot be discovered from Typer help;
- Renaming an internal field directly breaks user commands.

Open-ended overrides are valuable, but they must target the public schema rather than directly exposing the internal structure of `model_dump()`.

## 3.6 Router Converts Structured Configuration Back into Strings

This involves two YAML files with different purposes:

1. **Router launcher YAML**: Describes how many complete workers to launch and which GPUs and ports to allocate to them;
2. **Pipeline YAML**: Describes which Pipeline, stage topology, and runtime parameters are used inside each worker.

Current Router launcher YAML cannot describe a worker's Pipeline configuration structurally; it can only write worker arguments as a string:

```yaml
launcher:
  model_path: Qwen/Qwen3-Omni
  num_workers: 2
  worker_extra_args: "--config examples/configs/qwen.yaml --colocate"
```

Router first generates a base command for a worker:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni \
  --host 127.0.0.1 \
  --port 8011
```

At this point, `worker_extra_args` is still an ordinary string. Router uses `shlex.split()` to split it into:

```python
["--config", "examples/configs/qwen.yaml", "--colocate"]
```

It then appends these arguments to the base command:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni \
  --host 127.0.0.1 \
  --port 8011 \
  --config examples/configs/qwen.yaml \
  --colocate
```

When Router starts this worker subprocess, it also sets the physical GPU through an environment variable:

```bash
CUDA_VISIBLE_DEVICES=0
```

Only after the worker starts does `sgl-omni serve` parse `--config`, after which `ConfigManager` loads the second Pipeline YAML. Therefore, the actual flow is:

```text
Router launcher YAML
→ Read the worker_extra_args string
→ shlex.split into worker CLI arguments
→ Start the sgl-omni serve subprocess
→ Worker parses --config
→ ConfigManager loads Pipeline YAML
```

The problem is not that Router starts workers, but that Router only knows a string and does not understand what it configures. Field types, duplicate options, Pipeline file paths, and parameter compatibility can only be validated after the worker starts; the Router schema cannot describe or check the worker's actual configuration in advance.

## 3.7 Final Configuration Lacks Provenance

The current debug merged config only displays the final `PipelineConfig` and cannot answer:

- Whether a value came from a model default, YAML, or CLI;
- Which previous value it overrode;
- Whether it was set through a deprecated path;
- Whether it will later be modified by factory defaults or `ServerArgs.override()`.

When multiple entry points conflict, maintainers can only work backward through the helper order in `serve()`.

# 4. Design Principles and Non-Goals

## 4.1 How the Three Core Goals Are Implemented

| Core goal | Design mechanism | Acceptance criteria |
|---|---|---|
| Unify configuration processing code | canonical schema, `ConfigPatch`, single `ConfigResolver` | YAML, `--set`, and migration-period CLI aliases produce the same resolved value for the same field, with no entry-point-specific mutation helpers remaining |
| Clarify the developer extension interface | public typed schema, model extension schema, schema metadata generating CLI/documentation/tests | A new public field is defined only once; a new model field can only enter a registered extension schema |
| Clarify how users modify configuration | YAML/CLI responsibility boundary, name-keyed paths, `config schema/validate/resolve/explain/migrate` | Users can discover valid paths, validate configuration before startup, and see final values and override sources |

All subsequent data structures, migration steps, and tests must serve these three goals. A design that only reduces internal code without reducing the decision burden for developers or users is not considered completion of this refactoring.

## 4.2 Design Principles

1. **One semantic, one canonical path**: The same runtime feature must not expose both typed and untyped public paths.
2. **Input adapters contain no business logic**: YAML, CLI, and Router only produce typed assignments/patches.
3. **Resolve once, validate fully once**: Configuration is frozen after entering the runtime planner.
4. **Separate user configuration from derived plans**: GPU logical placement and the endpoint allocation root `endpoints.base_path` can be configured; runtime values such as `gpu_id`, a concrete socket endpoint, and NCCL port cannot.
5. **Explicit overrides with retained sources**: Cross-layer overrides are allowed but must be explainable; equal-specificity duplicate definitions within the same layer fail immediately.
6. **The compatibility layer has an explicit removal point**: Old interfaces may be adapted, but cannot permanently become a second canonical schema.

## 4.3 Non-Goals

- Do not copy all of SGLang's `ServerArgs` unchanged into Omni's top-level CLI;
- Do not allow YAML to select transports such as ZMQ/CUDA IPC that are derived from locality/placement;
- Do not change the runtime semantics of Stage/process/TP in this refactoring;
- Do not allow request-time parameters into startup-time `PipelineConfig`;
- Do not require Router and workers to share the same top-level schema; they share only the patch representation.

# 5. New Design Overview

The new design adds a unidirectional configuration compilation stage:

```text
Input documents
→ adapters
→ ConfigPatchSet
→ ConfigResolver
→ immutable ResolvedPipelineConfig
→ runtime planners
→ worker specs
```

| Order | Processing stage | Responsibility |
|---|---|---|
| 1 | Input adapters | Convert Python defaults, YAML, CLI, compatibility entry points, and Router into `ConfigPatch` objects |
| 2 | `ConfigPatchSet` | Store all modifications and their sources without executing entry-point-specific business logic |
| 3 | `ConfigResolver` | Uniformly handle types, precedence, duplicate definitions, conflicts, and full validation |
| 4 | `ResolvedPipelineConfig` | Generate a complete, read-only final user configuration with provenance |
| 5 | Runtime planners | Read only the final configuration and generate placement, process, and derived runtime plans |
| 6 | Worker specs | Construct cross-process startup parameters from the final configuration and plans |

All adapters depend only on the public schema and path registry. `ConfigResolver` is the only component that performs precedence, alias normalization, duplicate detection, and merging. The runtime planner no longer reads unresolved compatibility fields.

# 6. New User Interface

## 6.1 YAML Is the Persistent Declaration of a Pipeline

Current YAML can either list all stages in full or specify only modifications based on model defaults, but the file itself does not clearly indicate which form it uses. V2 uses `document_mode` to distinguish these two purposes.

### `full`: Complete Pipeline

`full` starts construction from an empty topology and does not inherit the default stages of a model configuration class. The file must fully declare the entry stage, all stages, and construction order, so the entire Pipeline can be determined from the file alone. The `example.*` factories below are used only to demonstrate the structure.

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

The names in `stage_order` must correspond one-to-one with those in `stages`; none may be missing, duplicated, or reference a nonexistent stage. `config export` outputs this complete form by default, making it suitable for archiving and exact reproduction.

### `partial`: Modify a Few Fields on a Model Template

`partial` must select a model template through `model.config` or a variant. The resolver first creates the template's default Pipeline and then applies the modifications written in YAML; stages and fields not written retain their model defaults.

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

This example means exactly three steps:

1. Create the default Pipeline for `Qwen3OmniSpeechColocatedPipelineConfig`;
2. Find the stage named `thinker`;
3. Change `max_running_requests` to 32.

`partial` can omit unchanged content, but can only modify stages that already exist in the template; it cannot create or delete stages. Use `full` when the complete topology must change.

### Why Stages Are Stored by Name

Current configuration stores stages as a list, so dynamic CLI may need to use unstable numeric positions:

```text
stages.4.runtime...
```

V2 stores stages as a name-keyed mapping:

```text
pipeline.stages.thinker.runtime...
```

The stage name becomes a stable identity, allowing YAML and CLI to use the same path without depending on a list index. The mapping itself does not express construction order, so a `full` document additionally uses `stage_order`; `partial` directly inherits the template's existing order.

The current `stage_overrides` exists because users need to modify a default stage list by stage name. In V2, `partial.pipeline.stages.<name>` is itself the canonical modification path, so a separate `stage_overrides` merge mechanism is no longer needed.

## 6.2 CLI Retains Only Startup Parameters and Generic Patches

The steady-state `serve` interface is divided into three groups:

### Configuration Selection

```text
--config FILE
```

or:

```text
--model-path MODEL [--variant text|speech|...]
```

The two modes are mutually exclusive. `--model-path` is shorthand for "construct configuration from model defaults"; when `--config` already contains `model.path`, it cannot be overridden.

### Service Process Parameters

The following values do not belong to the Pipeline and remain CLI options:

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

They are received separately by `ServerLaunchConfig` and are not written into `PipelineConfig`. If service configuration needs to be stored in the future, a separate deployment document should be introduced rather than putting it into Pipeline YAML.

### Temporary Pipeline Overrides

Uniformly use the repeatable:

```text
--set PATH=VALUE
```

For example:

```bash
sgl-omni serve \
  --config qwen.yaml \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=32 \
  --set pipeline.stages.thinker.placement.gpus='[0,1]' \
  --set pipeline.stages.thinker.placement.tp=2
```

The value is parsed as a YAML scalar/flow value, so booleans, nulls, numbers, and atomic list values share the same type rules. `--set` targets configurable leaves only; a list is one leaf and cannot be patched by index. Generic `--set` cannot replace a mapping/subtree; subtree replacement is available only in a `full` YAML document. The path must exist in the public schema; runtime-derived or deprecated internal paths cannot be written.

General stage arguments also support selector-based broadcast patches:

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

Selectors may use `capability`, `roles`, `stages`, and `exclude`; all supplied positive selectors must match, and `exclude` removes matches. The resolver expands each broadcast into concrete stage-leaf patches before duplicate checking and validation. Resolved storage remains only at `pipeline.stages.<stage>...`; no broadcast object survives in `ResolvedPipelineConfig`. Within one source, specificity is `capability broadcast < role/group selector < concrete stage`. Across different sources, the source layer always dominates specificity. Equal-specificity assignments to the same expanded leaf are errors.

The CLI equivalent is:

```bash
sgl-omni serve \
  --config qwen.yaml \
  --set-for capability=sglang runtime.sglang.max_running_requests=32 \
  --set pipeline.stages.thinker.runtime.sglang.max_running_requests=48
```

Here the concrete CLI assignment overrides the capability broadcast in the same CLI source. `config explain` shows every selector expansion and shadowed value; full export emits the expanded stage-local values, not selectors.

## 6.3 Final Role of Dedicated CLI Options

Existing dedicated runtime CLI options are retained during migration but no longer execute helpers. Each option declares only one canonical path:

```python
CliAlias(
    option="--thinker-max-running-requests",
    path="pipeline.stages.${role:thinker}.runtime.sglang.max_running_requests",
)
```

Parsing produces ordinary patches. Migration aliases have explicit specificity within the CLI source: broadcast/global alias < role alias < explicit concrete `--set`. A role alias overriding a broadcast alias is compatibility-preserving, not a duplicate; the shadowed broadcast value remains in provenance. Two assignments of the same specificity to the same canonical leaf remain errors. Source-layer precedence still dominates this rule.

After V2 stabilizes, dedicated runtime CLI options are removed in batches, leaving only `--set` and service process parameters. This prevents the final interface from permanently maintaining two definitions—"YAML field + dedicated CLI field." If the project decides to retain a very small number of frequently used aliases, their help, types, and patches must be generated automatically from schema metadata; they may not have independent business logic.

## 6.4 Configuration Tools

Add:

```bash
sgl-omni config schema
sgl-omni config validate --config FILE
sgl-omni config resolve --config FILE [--set ...]
sgl-omni config explain --config FILE PATH [--set ...]
sgl-omni config export --config FILE --format full
sgl-omni config migrate --input v1.yaml --output v2.yaml
```

`resolve` outputs the complete canonical Pipeline before freezing; `explain` outputs the final value, source chain, and whether a deprecated adapter was used. For example:

```text
pipeline.stages.thinker.runtime.sglang.max_running_requests = 32

1. model-default:Qwen3OmniSpeechPipelineConfig  64
2. file:qwen.yaml:41                           48
3. cli:--set[0]                                32
```

## 6.5 User Configuration Decisions

Users only need to select an entry point according to the following rules:

1. Write Pipeline changes that must be stored, reviewed, or reproduced into YAML canonical paths;
2. Use `--set` with the same canonical path for a temporary override of a Pipeline field in the current launch only;
3. Use service process CLI options to modify host, port, logging, or API policy;
4. If the field path is unknown, run `config schema` first; if the final value is uncertain, run `config resolve` or `config explain`;
5. Do not directly set `factory_args`, `runtime_overrides`, or runtime-derived fields.

Documentation and error messages therefore no longer need to answer "which of five entry points should be used for this value," but only "which canonical path does this value belong to, and should this change be persistent or temporary?"

# 7. Core Internal Data Structures

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

The initial patch protocol includes only `set` and `declare_stage`:

- `set` sets a typed value; `null` is always an ordinary value used to clear an optional field that permits null;
- `declare_stage` is produced only by the YAML adapter for `document_mode: full` and carries a typed declaration payload. `name`, `kind`, and `factory` are required; `extension_schema_id` is present when the stage uses a model extension. `kind` selects the legal runtime namespace and capability rules, so a declaration cannot first create an untyped stage and attach backend fields later.

A full document starts from an empty topology and can declare stages; a partial document and CLI `--set` can only modify existing stages and cannot create or delete stages. The names in `stage_order` must correspond one-to-one with the finally declared stages; none may be missing, duplicated, or reference an unknown stage. The initial patch protocol does not provide an operation for deleting inherited stages or mapping entries; use a full document to change stage membership.

The initial patch protocol always replaces lists as a whole; it does not add order-sensitive operations such as appending or removing items. Complex lists should preferably be written in YAML.

`ConfigPath` is not an arbitrary string traverser, but a path compiled from the public schema:

- Supports name-keyed stages;
- Validates whether a field is public/configurable;
- Returns the target Pydantic type;
- Prohibits access to `factory_args`, derived plans, and private fields;
- Lists nearby valid paths in errors.

## 7.2 `ConfigPatchSet`

Adapters output an ordered patch set. Layers are fixed as:

| Layer | Source |
|---|---|
| 0 | framework/schema defaults |
| 10 | model Python defaults |
| 20 | selected profile/variant |
| 30 | user YAML |
| 40 | Router structured worker patch |
| 50 | worker CLI `--set` or migration-period alias |

Layers resolve only cross-source precedence, not ambiguity within a layer:

- The same normalized path appears repeatedly at the same specificity in one document/CLI: error;
- Multiple sources in the same layer and at the same specificity write the same path: error, requiring the caller to define an explicit order or merge the documents;
- A higher layer overrides a lower layer: allowed and recorded in provenance;
- Compatibility aliases in one CLI source follow broadcast < role < explicit concrete `--set`; a more-specific value shadows a less-specific value with provenance, while equal-specificity duplicates are errors;
- Within one Router source, `worker_defaults` < a concrete per-worker entry; an entry may shadow a default with provenance, while duplicates within the same scope remain errors;
- Parent subtree and child leaf assignments overlap in the same layer: error;
- Across layers, apply the lower layer first: a higher-layer child can override the corresponding leaf of a lower-layer parent, while a higher-layer parent replaces the lower-layer subtree as a whole;
- A list is always treated as a leaf and is not merged by index;
- Generic `--set` can address only configurable leaves; mapping/subtree replacement requires full YAML.

The YAML parser must enable duplicate-key rejection and retain file, line, and column information for every mapping/scalar. Duplicate YAML keys, paths that duplicate after normalization at equal specificity, and same-specificity alias/canonical duplicates must be reported while generating the `ConfigPatchSet`; they must not first be read into an ordinary dict where a later value silently overwrites an earlier one.

## 7.3 `ConfigResolver`

The resolver order is fixed:

1. Collect adapter patches;
2. Normalize V1 paths/aliases into canonical paths;
3. Expand selector broadcasts into concrete stage-leaf patches and record expansion provenance;
4. Perform specificity, duplicate, and subtree overlap checks with layers taken into account;
5. Parse values according to the target schema;
6. Apply them to an immutable base tree by layer;
7. Construct the complete `ResolvedPipelineConfig`;
8. Perform cross-field and model capability validation;
9. Freeze the configuration and output the provenance map.

The resolver does not perform GPU topology probes, import stage factories, or construct `ServerArgs`. These belong to subsequent planners/workers, which can only read already-resolved typed fields.

## 7.4 `ResolvedPipelineConfig`

The resolved config is separate from the user document:

| Object | Content | User-settable |
|---|---|---|
| `PipelineConfigDocumentV2` | partial/full public document | Yes |
| `ResolvedPipelineConfig` | Complete typed config after applying all defaults and patches | Generated through the resolver |
| `StagePlacementPlan` | Actual GPU placement and memory accounting | No |
| `ProcessTopologyPlan` | OS process membership and TP ranks | No |
| `DerivedRuntimePlan` | Effective runtime values determinable by main-process topology/hardware probes | No |
| `StageWorkerProcessSpec` | spawn payload, endpoints, queues, factory defaults | No |
| `WorkerRuntimeDerivation` | Effective values and provenance determinable only during worker initialization | No |
| `ServerArgs` / `ModelConfig` | SGLang runtime objects inside the worker | No |

`ResolvedPipelineConfig` uses a frozen model, and runtime planners may not modify it in reverse. The resolver freezes only user intent, such as `custom_all_reduce: auto|on|off`, and may not depend on GPU topology or hardware probes.

All topology-dependent effective values determinable in the main process are written by runtime planners into a complete `DerivedRuntimePlan`, with topology/hardware sources recorded; the worker spec carries the slice of the plan needed by the current worker, and the worker builder may only consume it, not supplement or modify that plan in reverse.

Values determinable only after model loading or worker initialization enter a separate `WorkerRuntimeDerivation`, such as compatibility adjustments that depend on the actually loaded backend. It records provenance through worker startup diagnostics and `ServerArgs.override()` with a source, and does not write back into `DerivedRuntimePlan` or `ResolvedPipelineConfig`.

`config explain` explains user configuration and input provenance. Runtime-planner diagnostics display `DerivedRuntimePlan`, while worker startup diagnostics display `WorkerRuntimeDerivation`; together they explain the final effective value.

## 7.5 Canonical Runtime Namespace

Public runtime parameters enter typed namespaces:

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

The two environment paths store configured defaults only. Stage values override Pipeline defaults during configured-default resolution, while the inherited process environment still wins when `ResolvedProcessEnv` is derived at spawn. They do not absorb arbitrary `os.environ` reads into configuration precedence.

The following fields are no longer part of the public user interface:

```text
factory_args.server_args_overrides
runtime_overrides
runtime_arg_map
gpu_id
process_total_gpu_memory_fraction
nccl_port
concrete socket/IPC endpoints
```

`pipeline.endpoints.base_path` remains a user-configurable allocation policy input. The planner derives concrete socket paths/endpoints beneath it; only those concrete derived values are forbidden as user input.

`runtime.sglang.upstream_args` is a stage-local, validated escape hatch for SGLang fields not yet promoted into Omni's typed schema. Its schema is generated from the exact pinned upstream `ServerArgs` dataclass/schema and carries both the upstream version and a deterministic schema fingerprint. It rejects unknown fields and a maintained denylist of Omni-owned, derived, or unsafe fields (including device/process/port/endpoint ownership). A field already exposed as a typed Omni field is also rejected under `upstream_args`, preventing duplicate paths. Every accepted leaf retains provenance, and the fully assembled object still runs final upstream `ServerArgs` validation. Thus the escape hatch preserves **one semantic, one path** rather than creating an unvalidated second override dict. A stage whose `kind`/capabilities are not SGLang-backed rejects `runtime.sglang` entirely.

`encoder_mem_reserve` is a canonical SGLang-stage policy field rather than a factory argument. It is valid only when the effective thinker `mem_fraction_static` remains `auto`/unset and has not been pinned through that typed canonical path by any user source, including YAML, Router, selector expansion, or an alias. Because `mem_fraction_static` is already an Omni typed field, it is forbidden under `upstream_args` in the first place. The resolver uses provenance—not merely the final numeric value—to enforce this constraint. Runtime derivation then subtracts the reserve from the automatic thinker budget and applies the existing safety floor; it does not mutate the frozen policy or silently override a pinned fraction.

Model-specific parameters that have not yet entered the public schema are placed under:

```yaml
pipeline:
  stages:
    talker_ar:
      extensions:
        qwen3_omni:
          partial_start: true
```

An extension model must still register a Pydantic schema through the model package; a bare `dict[str, Any]` may not override public fields. Mature cross-model features are later promoted from extensions into the public runtime namespace.

Full export must materialize every effective behavior-defining typed or extension value, including values currently hidden in `factory_args` or Python factory defaults. For example, Qwen's effective `talker_max_seq_len` is `32768` in the model configuration even though the factory signature defaults to `4096`; the exported V2 document must contain `32768`. A model cannot opt in to V2 until all such behavior-defining factory defaults are represented in a public or registered extension schema and therefore survive export/import without consulting hidden Python defaults.

## 7.6 Relationship Between Roles and Stages

Existing CLI maintains separate thinker, talker, generation, and memory role maps through multiple class methods. V2 declares them uniformly in model config:

```yaml
pipeline:
  roles:
    thinker: thinker
    talker: talker_ar
    generation: talker_ar
    code2wav: code2wav
```

Roles are used only for:

- Migrating old typed CLI aliases;
- Model capability validation;
- Generating user-facing shortcuts/help.

Canonical storage always uses a concrete stage path. The resolver resolves roles when producing patches rather than repeatedly resolving them in runtime helpers.

## 7.7 The Single Process for Developers to Add Configuration

Developers first determine field ownership:

| New configuration type | Where it should be added | Where it should not be added |
|---|---|---|
| Pipeline/runtime field shared by multiple models | public typed schema | handwritten CLI helper, bare `factory_args` |
| Field specific to one model | model-registered extension schema | global `serve()` signature |
| HTTP server/process field | `ServerLaunchConfig` | `PipelineConfig` |
| Topology/hardware-derived value | `DerivedRuntimePlan` | user YAML/CLI |
| Value determinable only after worker initialization | `WorkerRuntimeDerivation` | resolver or Pipeline mutation |
| Request-time field | request protocol/schema | startup-time configuration |

The standard steps for adding a public typed field are fixed:

1. Define the path, type, default, description, and capability constraint in the canonical schema;
2. Consume the typed field in the resolved config → runtime consumer adapter;
3. Automatically generate JSON Schema, CLI path help, and YAML/CLI equivalence tests from schema metadata;
4. If compatibility with an old entry point is needed, add only a time-limited old path → canonical path mapping in `compat.py` and declare its removal Phase;
5. Do not add a parameter-specific mutation helper to `serve.py`.

Code review can use these rules to reject any new public configuration entry point that bypasses the canonical schema.

# 8. Responsibility Boundary Between YAML and CLI

The new interface establishes the following rules:

| Configuration category | YAML | CLI | Reason |
|---|---|---|---|
| Model, stage topology | canonical source | Select only through `--config`/`--model-path` | Structurally complex and must be persisted |
| GPU/TP/process placement | canonical source | Can be temporarily overridden with `--set` | Belongs to the Pipeline and requires unified validation |
| SGLang/runtime parameters | canonical source | Can be temporarily overridden with `--set` | Same path, same type, same validation |
| HTTP bind/log/API policy | Not included in Pipeline YAML | Dedicated CLI | Belongs to the service process, not the model Pipeline |
| Endpoint allocation policy (`endpoints.base_path`) | canonical source | Can be temporarily overridden with `--set` | User-configurable root for allocation |
| Concrete runtime-derived endpoint/port/gpu id | Not settable | Not settable | Owned by planner/runner |
| Router worker pool | Router YAML | Router CLI only selects source/routing parameters | Layered separately from the worker Pipeline |

CLI and YAML therefore still have two syntaxes, but only one set of Pipeline semantics:

```text
YAML leaf
→ ConfigPatch(path, value, file source)

CLI --set
→ ConfigPatch(path, value, cli source)
```

# 9. Router Interface Refactoring

## 9.1 Structured Worker Config

Launcher YAML removes the free-form string `worker_extra_args` and uses shared defaults plus explicit per-worker entries:

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
      gpus: ["0"]  # Intentional overlap with worker 0.
    - range: "2-3"
      gpus: ["GPU-3b6e...", "MIG-GPU-7d2a.../1/0"]
      capabilities: [chat]
```

`worker_defaults` supplies `config` or `model` (exactly one), service values, capabilities, and patches shared by workers. Each `workers` entry selects one worker with `index` or an inclusive set with `range`, and may override `gpus`, capabilities, and patches. Within one Router source, defaults are expanded first and then the concrete entry is applied; an entry patch may shadow the same path from defaults with provenance rather than being treated as a same-scope duplicate. Repetition within the entry itself remains an error. Repeated physical GPU sets are valid: the example deliberately gives workers 0 and 1 physical GPU `0`. Physical identifiers are preserved as written, including integer-like strings, UUIDs, and MIG identifiers; Router does not coerce them into logical IDs. Entries must cover each worker exactly once and overlapping index/range selectors are errors.

`num_gpus_per_worker` remains only as a fallback when no explicit worker `gpus` is provided; in that mode Router automatically and exclusively splits the available devices. It must not be combined with explicit per-worker GPU sets. After assigning a worker's actual visible set, Router validates Pipeline logical GPU IDs against `0..len(worker.gpus)-1`, not against a global count.

## 9.2 Worker Transport

The initial structured Router implementation may continue generating argv, but only as a machine interface:

```text
sgl-omni serve
  --config ...
  --config-patch-file /tmp/.../worker-0-patches.json
```

The patch file contains a serialized `ConfigPatchSet`, not a shell fragment. After validating the source and schema version, the worker passes it to the same resolver.

If Router and workers later switch to a Python API or supervisor protocol, they still reuse `ConfigPatchSet` without changing configuration semantics. GPU visibility continues to be managed by Router through the subprocess environment because it is a process resource boundary, not a Pipeline patch.

Router and workers use the same version of the schema registry and `ConfigResolver`. Before creating a subprocess, Router performs a complete dry-run resolve of each selected worker's `defaults + entry + config/model + patches`:

- Relative configuration paths are resolved relative to the directory containing the launcher YAML;
- Router must first resolve the selected model/config identity and load exactly the extension schemas referenced by that worker; inability to perform this lookup is a startup error;
- If a path, type, role, or extension cannot be resolved, fail before starting any worker;
- After receiving the patch file, the worker validates the schema version and content again, but may not use different merge rules.

GPU uses two explicit coordinate systems:

- Router is the sole component authorized to assign physical GPUs and establishes each worker's visible device set through `CUDA_VISIBLE_DEVICES`; explicit assignments may intentionally overlap across workers;
- Pipeline `placement.gpus` always uses worker-local logical GPU IDs, namely `0..N-1` after visible devices are renumbered.

Router dry-run must validate that every logical GPU ID is within `[0, len(actual worker visible GPU set))` and that TP size, GPU list, and worker allocation are consistent. Pipeline may not directly reference host physical GPU IDs. This prerequisite does not require Router to import or load every registered model—only the selected worker model and referenced extensions.

## 9.3 Precedence Between Router and Worker

The layers for managed workers are fixed:

```text
model defaults < worker Pipeline YAML < Router worker.patches < worker CLI emergency patch
```

`worker CLI emergency patch` is disallowed by default and enabled explicitly only in debug mode. Production configuration is determined entirely by Router YAML, preventing values generated by the supervisor from competing again with manually supplied argv.

# 10. Configuration Conflicts and Error Model

## 10.1 Conflict Categories

| Conflict type | Example | V2 behavior |
|---|---|---|
| Duplicate in the same source | Two `--set` options write the same path | Error before startup |
| alias specificity | broadcast alias and role alias target the same leaf | More-specific role value wins and shadowed value is recorded |
| same-specificity alias/canonical duplicate | two role aliases, or two explicit `--set` values, write the same leaf | Error before startup |
| typed/extension overlap | An extension attempts to set a public SGLang field | Schema registration fails |
| parent/child overlap | Two patch sources in the same layer set `runtime.sglang` and one of its children | Error in the same layer; allowed and recorded across layers |
| Ownership violation | User sets `gpu_id` or a concrete endpoint | Path is not public; parsing fails; `endpoints.base_path` remains configurable |
| Unsupported capability | A non-SGLang stage sets `runtime.sglang` | Model validation fails |
| Cross-field inconsistency | TP=2 but only one GPU exists | Resolved model validation fails |
| deprecated-document/canonical duplicate | V1 `runtime_overrides` and a V2 field in the same document layer | Migration error; CLI aliases instead follow the explicit specificity rule |

## 10.2 Atomicity

The resolver applies all patches to an immutable in-memory tree. If any path, type, conflict, or validation fails, it returns no partial config and does not enter the runtime planner.

Errors must include:

- canonical path;
- source location;
- previous value and its source;
- conflict rule;
- actionable remediation.

For example:

```text
duplicate configuration for
pipeline.stages.thinker.runtime.sglang.mem_fraction_static

cli:--thinker-mem-fraction-static = 0.70
cli:--generation-mem-fraction-static = 0.65

The two role aliases have equal specificity and resolve to the same stage.
Remove one alias or use an explicit concrete `--set`.
```

# 11. Configuration Compilation and Runtime Boundary

The V2 compilation chain is:

```text
① Parse input documents
→ ② Normalize patches
→ ③ Resolve + validate + freeze
→ ④ Build placement/process plans
→ ⑤ Build derived runtime plan
→ ⑥ Build worker specs
→ ⑦ Construct worker runtime objects
```

Read it strictly as **① → ② → ③ → ④ → ⑤ → ⑥ → ⑦**:

- After ②, all inputs use canonical paths;
- ③ is the last step that modifies user configuration;
- ④ only generates derived plans and does not write back to the Pipeline;
- ⑤ stores effective runtime values determinable by main-process topology/hardware probes;
- ⑥ resolves signature-independent factory arguments in the parent and assembles them into cross-process DTOs;
- ⑦ imports the factory, injects only missing signature-dependent defaults, and then constructs `ServerArgs` and `ModelConfig`; adjustments determinable only at this stage are recorded in `WorkerRuntimeDerivation`.

The current `_apply_tensor_parallel_server_args_overrides()` modifies stage server args based on a topology probe. V2 splits this into:

1. User configuration expresses `custom_all_reduce: auto|on|off`;
2. The placement planner produces topology capabilities;
3. The main-process runtime planner produces `DerivedRuntimePlan` from both;
4. The worker builder only consumes the effective `ServerArgs` value in the plan;
5. It does not modify the frozen Pipeline or repeat the same topology decision in the worker.

Other topology-dependent parameters follow the same pattern.

# 12. Scope of Changes

## 12.1 Configuration Schema and Resolver

| Module | Change |
|---|---|
| `sglang_omni/config/schema.py` | Add V2 public schema, name-keyed stages, typed runtime namespaces, frozen resolved model |
| `sglang_omni/config/patch.py` (new) | `ConfigPath`, `ConfigSource`, `ConfigPatch`, duplicate/overlap checks |
| `sglang_omni/config/resolver.py` (new) | selector expansion, specificity/layer merge, type conversion, provenance, full validation |
| `sglang_omni/config/compat.py` (new) | Time-limited adapter from V1 YAML/dotted/typed CLI to V2 patches |
| `sglang_omni/config/sglang_schema.py` (new) | Generate and fingerprint the exact pinned `ServerArgs` passthrough schema and enforce the ownership/unsafe denylist |
| `sglang_omni/models/registry.py` | Register V2 model defaults, roles, stage kinds/capabilities, and extension schemas |

The file-loading responsibility of `ConfigManager` is moved into the adapter/resolver. The old class remains as a facade during migration, and its custom dotted merge is ultimately removed.

## 12.2 CLI

| Module | Change |
|---|---|
| `sglang_omni/cli/serve.py` | After parsing parameters, construct only `ServerLaunchConfig` and patches; remove business mutation helpers |
| `sglang_omni/cli/config.py` | Add schema/validate/resolve/explain/migrate/export, selector expansion display, and environment diagnostics |
| `sglang_omni/cli/__init__.py` | Disable unknown options; all open-ended overrides use explicit `--set` |

During migration, typed CLI aliases may continue to appear in the `serve()` signature, but the alias registry automatically generates their patch implementation.

## 12.3 Runtime

| Module | Change |
|---|---|
| `sglang_omni/config/runtime.py` | Convert typed resolved runtime into parent-resolved static factory kwargs; no longer resolve conflicts between user sources |
| `sglang_omni/pipeline/runtime_config.py` | Read only frozen resolved config and output placement/process and `DerivedRuntimePlan` |
| `sglang_omni/pipeline/mp_runner.py` | Construct specs and `ResolvedProcessEnv` from resolved config/plans; preserve the static-args/signature-injection boundary |
| `sglang_omni/pipeline/stage_workers.py` | Apply configured environment defaults without overriding inherited values and emit process-environment diagnostics |
| `sglang_omni/scheduling/sglang_backend/server_args_builder.py` | Construct `ServerArgs` from typed SGLang config, retaining explicit derived runtime overrides |

Model factories may temporarily continue receiving `server_args_overrides`, but this dict can only be generated internally by the typed runtime adapter and is no longer a public YAML/CLI entry point.

## 12.4 Router

| Module | Change |
|---|---|
| `sglang_omni_router/launcher/config.py` | `worker_defaults` and index/range-selected `workers` schema, capabilities, physical GPU identifiers, and patches |
| `sglang_omni_router/launcher/local.py` | Generate patch files instead of concatenating `worker_extra_args` |
| `sglang_omni_router/serve.py` | Worker source validation and structured launcher integration |

# 13. Phased Compatibility Plan

V2 uses a phased migration. It does not require all configuration to be modified in one release, but every deprecated entry point has a removal phase.

## 13.1 Phase 0: Establish the Resolver Without Changing Default Behavior

- First freeze an independent oracle by running the untouched latest-main V1 implementation at `2b45073c` and recording its final outputs: resolved Pipeline shape, factory kwargs, placement/process plans, worker specs, and user-controlled `ServerArgs` fields. These legacy goldens are captured before routing V1 through any new adapter or resolver;
- Add read-only provenance reconstruction plus `config resolve/explain` around current behavior, without making the new resolver authoritative;
- Introduce `ConfigPatch`, provenance, and `config resolve/explain`;
- In shadow mode, current model defaults, V1 YAML, dotted CLI, and typed CLI also produce patches through compatibility adapters, without replacing the V1 execution path;
- Compare adapter/resolver output against the independent untouched-V1 oracle; V1-adapter-versus-V2-through-the-same-resolver equivalence is supplementary and cannot replace that oracle;
- Runtime continues consuming the existing `PipelineConfig` shape;
- Duplicate V1 sources newly discovered by the resolver initially record diagnostics without changing current success/failure outcomes;
- V2 inputs in internal tests follow strict duplicate/overlap rules from the beginning.

The goal of this phase is to first obtain a single merge engine, not immediately change user syntax or validation behavior. Starting in Phase 1, V2 inputs fail strictly while V1 inputs warn about newly discovered duplicate sources; starting in Phase 2, semantically duplicate V1 inputs also fail.

## 13.2 Phase 1: V2 Opt-In

- Support `schema_version: 2`;
- New documentation and exports display V2 by default, with a flag to retain V1 export;
- Router supports structured worker configuration while remaining compatible with `worker_extra_args`;
- Typed CLI aliases print the equivalent canonical `--set`;
- Telemetry/logging counts deprecated-path usage without uploading actual configuration values.

V2 documents may not contain V1 `stage_overrides`, `runtime_overrides`, or public `factory_args.server_args_overrides`.
V1 inputs retain their current final-value semantics, but newly discovered duplicate sources produce warnings.
A model is not eligible for V2 opt-in until all behavior-defining values hidden in its `factory_args` or factory defaults are represented by public or registered extension schema and round-trip through full export.

## 13.3 Phase 2: V2 by Default

- `config export`, official examples, cookbooks, and Router examples all switch to V2;
- YAML without `schema_version` is treated as V1 and produces a warning;
- Unknown dotted options are disabled, requiring explicit `--set`;
- Runtime typed CLI aliases produce deprecation warnings;
- Semantically duplicate V1 deprecated/canonical inputs become startup errors;
- `worker_extra_args` produces a warning, and `config migrate` automatically converts recognizable parameters.

## 13.4 Phase 3: Remove Old Entry Points

At the next explicitly announced major/minor compatibility boundary:

- Remove `stage_overrides`;
- Remove public `runtime_overrides`;
- Prohibit YAML from setting public `factory_args.server_args_overrides`;
- Remove dynamic unknown-option dotted CLI;
- Remove dedicated runtime typed CLI aliases;
- Remove Router `worker_extra_args`;
- Remove V1 path traversal from `ConfigManager.merge_config()`.

When reading V1 YAML, provide a clear error and an offline migration command rather than continuing implicit compatibility in the server.

## 13.5 Compatibility Behavior Table

| Current interface | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| V1 Pipeline YAML | Supported | Supported + migratable | warning | Rejected |
| `stage_overrides` | Supported | warning | warning | Rejected |
| `runtime_overrides` | Supported | warning | warning | Rejected |
| dotted unknown CLI | Supported | warning | Reject with `--set` guidance | Removed |
| runtime typed CLI | Supported | warning | warning | Removed |
| V2 YAML | Internal tests | opt-in | Default | Only format |
| Router `worker_extra_args` | Supported | warning | warning | Rejected |
| structured Router worker | Internal tests | opt-in | Default | Only format |

# 14. Validation and Testing

Validation is divided into resolver and runtime planning layers. The former proves that "the user declaration is unique and correctly typed," while the latter proves that "the resolved topology is executable."

## 14.1 Resolver Unit Tests

| Test category | Checks |
|---|---|
| path | Valid field, unknown field, private/derived field, stage name |
| type | scalar, null, list, dict, enum, invalid coercion |
| document structure | duplicate YAML key, full/partial mode, stage declaration, `stage_order` completeness |
| precedence | defaults, profile, YAML, Router patch, CLI, source layer dominating selector specificity |
| selectors | capability/role/stage/exclude matching, expansion, empty selection, concrete stage storage/export |
| duplicates | same source/specificity, same layer, alias specificity, parent/child |
| cross-layer subtree | higher-layer child overrides lower-layer parent, higher-layer parent replaces lower-layer subtree, whole-list replacement |
| provenance | previous value, source, location, and deprecated marker for each override |
| atomicity | No partial resolved config is produced if any patch fails |
| frozen config | Assignment is prohibited after the resolver returns |
| role | Unique role-to-stage resolution, unknown/unsupported role |
| extension | Registered schema, unknown key, prohibition on overriding public namespace, full-export round trip of hidden defaults |
| upstream args | exact pinned `ServerArgs` version/fingerprint, unknown fields, denylist, typed-field duplication, per-leaf provenance, final `ServerArgs` validation |
| environment | `env_defaults`/stage `env` merge, inherited-environment non-override, derived `ResolvedProcessEnv`, external-read diagnostics |

## 14.2 CLI/YAML Equivalence

Create parameterized tests for every public typed field:

```text
YAML canonical leaf
==
CLI --set canonical path
==
migration-period typed CLI alias
```

All three inputs must produce the same `ResolvedPipelineConfig`, differing only in provenance. The parameter table is generated automatically from schema metadata to avoid handwritten tests omitting new fields.

## 14.3 V1 Migration Golden Tests

Select official examples:

- Qwen3-Omni text/speech/colocated;
- Qwen3-TTS;
- Higgs, MOSS, FishAudio;
- Single-stage ASR;
- Ming full-stages YAML;
- TP and process isolation configurations.

For each case, first capture the output of the untouched latest-main V1 implementation as an independent golden. Then compare both the V1 adapter and migrated V2 document against that oracle:

- Stage topology and order;
- placement/TP/process;
- typed runtime;
- Final factory kwargs;
- placement/process plans;
- worker specs;
- User-controllable `ServerArgs` fields.

Use structural comparison for runtime-derived endpoints and temporary ports rather than comparing random values.

## 14.4 Conflict Regression

The following must be covered:

- typed `mem_fraction_static` and old `server_args_overrides`;
- global and per-role CLI;
- broadcast alias, role alias, and explicit `--set` specificity, including shadowed provenance;
- selector broadcast and concrete-stage override;
- `--encoder-mem-reserve` and explicit fraction;
- `tp_size` and GPU list;
- generation role and thinker role targeting the same/different stages;
- deprecated path and canonical path;
- Router patch and worker CLI;
- extension and public namespace.

Every conflict assertion checks that the canonical path and both sources appear in the error.

## 14.5 Runtime Boundary

Verify:

- The runtime planner does not modify frozen config;
- Topology-dependent effective values enter only `DerivedRuntimePlan`;
- The worker builder only consumes `DerivedRuntimePlan` and cannot supplement or repeat decisions;
- Worker-only effective values enter only `WorkerRuntimeDerivation` and retain their sources;
- Child processes do not resolve user configuration again;
- Factory signature injection supplies only missing derived defaults;
- Users cannot set `gpu_id`, cumulative process budget, NCCL port, or a concrete endpoint, while `endpoints.base_path` remains configurable;
- TP environment mapping matches current behavior;
- Worker specs for the no-config/default model path match golden files;
- `ServerArgs.override()` continues recording the source of worker-internal derived mutations.

## 14.6 Router

Verify:

- Worker commands generated from structured configuration contain no shell-like extra args;
- Patch files are independent per worker and use the correct schema version;
- model/config are mutually exclusive;
- Router uses the same registry/resolver version and completes a dry-run before starting subprocesses;
- Launcher-relative config paths and model extension schemas resolve correctly;
- Router YAML path/type/role/extension failures occur before starting subprocesses;
- GPUs continue to be passed through the environment;
- Physical GPU assignment is correctly converted to worker-local logical GPU IDs;
- Index/range coverage, per-worker capabilities/patches, and selected-model/extension lookup are validated;
- `worker_defaults` < per-worker entry shadowing retains provenance, while duplicates within each scope are rejected;
- Repeated physical GPU sets across workers are preserved, including two workers sharing GPU `0`;
- Integer-like, UUID, and MIG physical identifiers round-trip unchanged;
- Out-of-range logical GPUs are checked against each actual visible set, and TP/allocation inconsistencies fail before startup;
- Exclusive `num_gpus_per_worker` splitting is used only when explicit `gpus` are absent;
- V1 `worker_extra_args` migration recognizes `--config`, typed aliases, and `--set`;
- Arbitrary shell tokens that cannot be migrated safely fail explicitly and are not silently discarded.

## 14.7 Phase Admission Criteria

Every migration Phase must have compatibility matrix tests asserting, for each old entry point:

- success, warning, or error;
- Final resolved value;
- provenance/diagnostic;
- Recommended migration command.

Before entering the next Phase, all of the following must be satisfied:

1. All official Pipeline and Router examples pass migration golden tests;
2. Structured Router has completed at least one multi-worker E2E;
3. Runtime no longer directly reads public compatibility fields;
4. The warning/error and full compatibility matrices for the current Phase are fixed in CI;
5. All official examples, documentation, tests, and CI invocations have migrated;
6. Release notes have announced the removal for a predetermined `N` releases;
7. Those `N` announced releases have shipped before advancing to the removal Phase.

Repository state and release history are the decidable gates. Local or explicitly opt-in telemetry may diagnose remaining use, but no unobservable global deprecated-usage threshold can block or authorize a Phase transition.

# 15. Release and Documentation

When releasing V2, provide all of the following:

1. V1 → V2 migration guide;
2. CLI/YAML responsibility table;
3. Canonical path reference and generated JSON Schema;
4. Debugging examples for `config resolve/explain`;
5. Removal versions for deprecated entry points;
6. V1/V2 diffs for official examples;
7. Router structured worker example.

CLI `--help` no longer lists all model runtime fields; it only describes `--set` and points to:

```bash
sgl-omni config schema
sgl-omni config resolve
```

Model-specific fields are displayed dynamically by the selected model config/extension schema rather than fixing every model option into one global `serve()` signature.

# 16. Expected Result

The stable relationship after refactoring is:

```text
YAML / CLI / Router
        ↓
The same ConfigPatch type
        ↓
The single ConfigResolver
        ↓
Frozen ResolvedPipelineConfig
        ↓
Read-only runtime planners
```

YAML and CLI retain different use cases:

- YAML stores a complete, reviewable, reproducible Pipeline;
- CLI selects configuration, sets service process parameters, and uses `--set` for temporary Pipeline patches;
- Router YAML structurally references worker config and patches.

They no longer define runtime parameter types, defaults, precedence, and mutation logic separately. When adding a shared configuration field, it only needs to be defined once in the canonical schema; CLI, JSON Schema, documentation, patch types, and equivalence tests are all generated from that definition.
