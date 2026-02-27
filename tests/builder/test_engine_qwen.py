"""Engine tests for the Qwen family plugin."""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class QwenPluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.qwen"
    model_type = "qwen3"


class TestQwenEngine(FamilyPluginTestMixin):
    tester_class = QwenPluginTester
