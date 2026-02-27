"""Engine tests for the Mistral family plugin."""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class MistralPluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.mistral"
    model_type = "mistral"


class TestMistralEngine(FamilyPluginTestMixin):
    tester_class = MistralPluginTester
