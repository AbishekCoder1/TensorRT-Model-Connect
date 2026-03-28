"""Domain types and protocol contracts for the unified E2E testing framework.

This module is the stable foundation that all other harness components depend on.
It contains only stdlib imports (dataclasses, enum, typing) and defines:

- Domain dataclasses: E2ECase, StageSpec, PreflightRequirement, ThresholdProfile,
  StageOutput, CompareResult, E2EResult, RunContext
- Enums: FailureType, OracleLevel, E2EStatus
- Protocols: TaskStrategyRunner, ReferenceBackendRunner, Comparator,
  ArtifactSink, DeterminismPolicy
- Constants: RUNTIME_TO_TASK_STRATEGY mapping

The orchestrator imports ONLY these types and protocols. Concrete
implementations live in strategy runners, reference backends, and comparators.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailureType(enum.Enum):
    """Classification of how an E2E run failed.

    Used in E2EResult to indicate the phase where failure occurred,
    enabling fast triage without reading full logs.
    """

    PRECHECK_FAIL = "precheck_fail"
    BUILD_FAIL = "build_fail"
    TRT_RUN_FAIL = "trt_run_fail"
    REFERENCE_RUN_FAIL = "reference_run_fail"
    COMPARE_FAIL = "compare_fail"
    DETERMINISM_FAIL = "determinism_fail"
    ARTIFACT_WRITE_FAIL = "artifact_write_fail"


class OracleLevel(enum.Enum):
    """Strength of the reference oracle used for comparison.

    Higher levels imply stronger parity guarantees. The level is recorded in
    result.json so that consumers know what a "pass" actually means.

    L1: External reference (HF Transformers / Diffusers / official lib).
    L2: Internal reference (custom torch / Python implementation).
    L3: Snapshot regression (golden outputs from a trusted prior run).
    L4: Invariants only (metamorphic / property-based checks, no reference).
    """

    L1_EXTERNAL_REFERENCE = "L1_external_reference"
    L2_INTERNAL_REFERENCE = "L2_internal_reference"
    L3_SNAPSHOT_REGRESSION = "L3_snapshot_regression"
    L4_INVARIANTS = "L4_invariants"


class E2EStatus(enum.Enum):
    """Terminal status of an E2E run."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class StageStatus(enum.Enum):
    """Status of a single stage comparison.

    PASSED: All metrics met their thresholds.
    FAILED: One or more gated metrics failed.
    SKIPPED: Stage was not compared (no reference/comparator available).
    ERROR: Stage comparison raised an exception.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreflightRequirement:
    """A single prerequisite that must be satisfied before a case can run.

    Attributes:
        kind: Requirement type identifier. Supported kinds:
            - "binary_exists": args must contain "path".
            - "gpu_memory_min_gb": args must contain "min_gb".
            - "hf_auth_token_present": no args needed.
            - "asset_exists": args must contain "path".
            - "python_module_available": args must contain "module".
        args: Parameters specific to the requirement kind.
        gating: If True (default), an unmet requirement causes PRECHECK_FAIL.
            If False, the requirement is advisory and logged but does not block.
    """

    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    gating: bool = True


@dataclass
class StageSpec:
    """Describes one stage in the E2E lifecycle for a model case.

    For simple models (text generation), there is typically one stage.
    For composite pipelines (diffusion, omni), there can be many.

    Attributes:
        name: Stage identifier (e.g. "generate", "t5_encode", "vae_decode").
        required: If True, stage failure causes overall case failure.
        runner_override: Optional strategy runner name to use instead of the
            default for this case's task_strategy.
        comparator_override: Optional comparator name to use instead of the
            default for this case's task_strategy.
    """

    name: str
    required: bool = True
    runner_override: Optional[str] = None
    comparator_override: Optional[str] = None


@dataclass
class ThresholdProfile:
    """Tolerance profile for comparing TRT vs reference outputs.

    Attributes:
        task_strategy: The task strategy this profile applies to.
        profile_name: Human-readable profile name (e.g. "fp16_default").
        metrics: Metric name -> threshold value mapping. Metric names are
            strategy-specific (e.g. "logit_cosine_p5", "token_agreement_rate",
            "mIoU", "mel_distance", "psnr").
        percentile_gates: Optional percentile-based gates (e.g.
            {"logit_rel_l2_p95": 0.01, "logit_rel_l2_p99": 0.05}).
        composite_rules: Optional list of composite gating rules that
            combine multiple metrics (e.g. "pass if token_agreement >= 0.9
            OR (cosine >= 0.999 AND topk_hit >= 0.95)").
    """

    task_strategy: str
    profile_name: str = "default"
    metrics: Dict[str, float] = field(default_factory=dict)
    percentile_gates: Dict[str, float] = field(default_factory=dict)
    composite_rules: List[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Self-describing result for a single comparison metric.

    Combines the metric value, its threshold, the comparison operator,
    and the pass/fail outcome into one structure.

    Attributes:
        value: Computed metric value.
        threshold: Threshold the value was compared against (None if informational).
        operator: Comparison operator as a string (e.g. ">=", "<=", "==").
        passed: Whether this metric met its threshold.
        note: Optional human-readable annotation.
    """

    value: float
    threshold: Optional[float] = None
    operator: str = ">="
    passed: bool = True
    note: str = ""


@dataclass
class E2ECase:
    """Complete definition of one E2E test case (one model).

    Loaded from a manifest v2 JSON file. Contains everything needed to
    build, run, compare, and report for a single model.

    Attributes:
        name: Unique case identifier (e.g. "qwen3-0.6b").
        hf_id: HuggingFace model ID or local path.
        family: Family plugin name (e.g. "qwen", "llama").
        runtime_strategy: C++ runtime backend selector (e.g. "decoder_kv_cache").
        task_strategy: Logical task type derived from runtime_strategy
            (e.g. "text_generation_causal"). See RUNTIME_TO_TASK_STRATEGY.
        reference_backend: Which reference implementation to compare against
            (e.g. "hf_transformers", "torch_reference", "golden_snapshot").
        oracle_level: Strength of the reference oracle. See OracleLevel.
        bundle: Bundle filename (e.g. "qwen3-0.6b.trtfb").
        inputs: Task-specific inputs (prompt, image path, audio path, etc.).
        preflight: List of prerequisites to check before running.
        stages: Ordered list of stages to execute.
        comparison_profile: Name of the ThresholdProfile to use.
        threshold_overrides: Per-metric overrides that take precedence over
            the profile defaults.
        determinism: Settings for determinism/reproducibility checks
            (e.g. {"reruns": 2, "seed": 42}).
        metadata: Arbitrary extra fields (notes, trust_remote_code, etc.).
    """

    name: str
    hf_id: str
    family: str
    runtime_strategy: str
    task_strategy: str = ""
    reference_backend: str = "hf_transformers"
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value
    bundle: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    preflight: List[PreflightRequirement] = field(default_factory=list)
    stages: List[StageSpec] = field(default_factory=list)
    comparison_profile: str = "default"
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
    determinism: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive task_strategy from runtime_strategy if not set."""
        if not self.task_strategy:
            self.task_strategy = RUNTIME_TO_TASK_STRATEGY.get(
                self.runtime_strategy, self.runtime_strategy
            )


@dataclass
class StageOutput:
    """Output produced by running one stage (TRT or reference).

    Intentionally flexible: different task strategies populate different
    fields. The ``data`` dict is the primary carrier for modality-specific
    outputs (logits array, image path, audio samples, embeddings, etc.).

    Attributes:
        stage_name: Which stage produced this output.
        data: Modality-specific output data. Keys depend on task_strategy.
            Examples:
            - text_generation: {"token_ids": [...], "logits_path": "/tmp/..."}
            - segmentation: {"mask": <np.ndarray>, "class_ids": [...]}
            - diffusion: {"frames_dir": "/tmp/...", "latents": <np.ndarray>}
        text: Generated text (for text-producing strategies).
        logits: Path to saved logits or in-memory array (optional).
        timing_s: Wall-clock time for this stage in seconds.
        metadata: Extra info (command used, return code, warnings, etc.).
    """

    stage_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    logits: Any = None  # numpy array or path string
    timing_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResult:
    """Outcome of comparing TRT output to reference output for one stage.

    Each metric is self-describing via :class:`MetricResult`, carrying its
    value, threshold, operator, and pass/fail in a single object.

    Attributes:
        stage_name: Which stage was compared.
        status: Stage-level status string (see :class:`StageStatus`).
        metrics: Per-metric results keyed by metric name.
        composite_rule: Human-readable description of the composite gating
            logic (e.g. "(cosine >= T OR rel_l2 <= T) AND agreement >= T").
        message: Summary message for display.
    """

    stage_name: str
    status: str = StageStatus.FAILED.value
    metrics: Dict[str, "MetricResult"] = field(default_factory=dict)
    composite_rule: str = ""
    message: str = ""

    @property
    def passed(self) -> bool:
        """Backward-compatible boolean: True when status is 'passed'."""
        return self.status == StageStatus.PASSED.value


@dataclass
class E2EResult:
    """Final structured result of an E2E test run for one model case.

    Serialized to a single consolidated ``result.json`` in the artifacts
    directory.  Contains everything previously spread across case.json,
    env_fingerprint.json, commands.json, and per-stage files.

    Attributes:
        case_name: The E2ECase.name this result belongs to.
        status: Terminal status (pass/fail/skip/error).
        failure_type: If status is fail/error, which phase failed.
        oracle_level: Oracle strength for this run (from the case).
        stages: Per-stage comparison results.
        determinism: Results of determinism reruns (if any).
        timing: Phase-level timing (build_s, trt_run_s, ref_run_s, etc.).
        env_fingerprint: Environment info (GPU, driver, TRT version, etc.).
        timestamp: ISO 8601 timestamp of when this result was produced.
        repro_commands: Shell commands to reproduce each phase of the test.
        case_config: Snapshot of the E2ECase configuration (was case.json).
        commands: Subprocess command log (was commands.json).
        stage_outputs: Per-stage TRT/ref output summaries (was stages/ dir).
        artifacts: Map of artifact type to relative path(s) within model dir.
        log_file: Relative path to the merged log file (e2e_run.log).
    """

    case_name: str
    status: str = E2EStatus.PASS.value
    failure_type: Optional[str] = None
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value
    stages: Dict[str, CompareResult] = field(default_factory=dict)
    determinism: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    env_fingerprint: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    repro_commands: Dict[str, str] = field(default_factory=dict)
    case_config: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    log_file: str = ""


@dataclass
class RunContext:
    """Runtime context passed to strategy runners and reference backends.

    Carries resolved paths, flags, and the case definition needed to
    execute a stage.

    Attributes:
        case: The E2ECase being executed.
        artifacts_dir: Directory for writing stage outputs and logs.
        binary_path: Path to the C++ trtf binary.
        hf_python: Path to the Python interpreter with HF tokenizers.
        ld_library_path: LD_LIBRARY_PATH with TRT/CUDA libs.
        engine_dir: Directory containing .trtfb bundles.
        rebuild: If True, force rebuild bundles from HF.
        verbose: If True, emit extra debug output.
    """

    case: E2ECase
    artifacts_dir: str = ""
    binary_path: str = ""
    hf_python: str = ""
    ld_library_path: str = ""
    engine_dir: str = ""
    rebuild: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskStrategyRunner(Protocol):
    """Executes the TRT inference path for a specific task strategy.

    Implementations handle the details of invoking the C++ binary or Python
    debug runner for their task type (text generation, VL, diffusion, etc.).
    """

    @property
    def strategy_name(self) -> str:
        """Unique identifier matching a task_strategy value."""
        ...

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Execute one stage and return its output."""
        ...


@runtime_checkable
class ReferenceBackendRunner(Protocol):
    """Executes the reference inference path for comparison.

    Implementations may use HF Transformers, Diffusers, custom torch code,
    or load golden snapshots.
    """

    @property
    def backend_name(self) -> str:
        """Unique identifier matching a reference_backend value."""
        ...

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Execute one reference stage and return its output."""
        ...


@runtime_checkable
class Comparator(Protocol):
    """Compares TRT output against reference output for a task strategy.

    Implementations compute task-appropriate metrics (cosine similarity for
    text, mIoU for segmentation, PSNR for diffusion, etc.) and gate on
    the provided threshold profile.
    """

    @property
    def task_strategy(self) -> str:
        """The task strategy this comparator handles."""
        ...

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare TRT vs reference outputs and return gated result."""
        ...


@runtime_checkable
class ArtifactSink(Protocol):
    """Persists commands, stage outputs, comparison results, and final report.

    Implementations write to disk (the common case) or could aggregate
    results in memory for testing.
    """

    def log_command(
        self, command: List[str], rc: int, stdout: str, stderr: str
    ) -> None:
        """Record a subprocess invocation and its output."""
        ...

    def write_stage_output(self, name: str, output: StageOutput) -> None:
        """Persist one stage's output to artifacts."""
        ...

    def write_compare(self, name: str, result: CompareResult) -> None:
        """Persist one stage's comparison result."""
        ...

    def register_artifact(self, key: str, rel_path: str) -> None:
        """Register a modality artifact (PNG, WAV, NPY) by key and path."""
        ...

    def finalize(self, result: E2EResult) -> str:
        """Write the final result.json and return its path."""
        ...


@runtime_checkable
class DeterminismPolicy(Protocol):
    """Checks TRT output reproducibility across multiple runs.

    Called with outputs from N reruns of the same stage. Returns a
    CompareResult indicating whether intra-run variance is acceptable.
    """

    def check(
        self, case: E2ECase, outputs: List[StageOutput]
    ) -> CompareResult:
        """Evaluate determinism across multiple stage outputs."""
        ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNTIME_TO_TASK_STRATEGY: Dict[str, str] = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_moe": "text_generation_causal",
    "ssm_recurrent": "text_generation_causal",
    "rwkv_recurrent": "text_generation_causal",
    "hybrid_mamba_attention": "text_generation_causal",
    "vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "text_to_audio": "text_to_audio",              # legacy alias
    "text_to_audio_bark": "text_to_audio",
    "text_to_audio_magpie": "text_to_audio",
    "speech_to_speech": "speech_to_speech",
    "segmentation": "segmentation",
    "prompted_segmentation": "prompted_segmentation",
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "diffusion": "diffusion_media_generation",      # legacy alias
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_zimage": "diffusion_media_generation",
    "diffusion_pixart": "diffusion_media_generation",
    "omni_multimodal": "omni_multimodal",
    "text_to_text": "text_generation_causal",
    "marian_translation": "text_generation_causal",
    "seq2seq_encoder_decoder": "text_generation_causal",
}
