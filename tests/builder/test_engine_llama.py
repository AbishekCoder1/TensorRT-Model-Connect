"""Engine tests for the LLaMA family plugin."""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class LlamaPluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.llama"
    model_type = "llama"


class TestLlamaEngine(FamilyPluginTestMixin):
    tester_class = LlamaPluginTester
