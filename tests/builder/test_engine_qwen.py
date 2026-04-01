"""Engine tests for the Qwen family plugin.

Trace: ARCH-FAM-001, UD-FAM-QWEN-01
Intent: Validate the Qwen family plugin weight loading and standard decoder key mapping with RMSNorm, SwiGLU MLP, and RoPE.
Preconditions: safetensors and trtf_build are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class QwenPluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.qwen"
    model_type = "qwen3"


class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester
