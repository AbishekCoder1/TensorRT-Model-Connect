"""Engine tests for the Granite family plugin."""
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class GranitePluginTester(FamilyPluginTester):
    plugin_module = "trtf_build.families.granite"
    model_type = "granite"


class TestGraniteEngine(FamilyPluginTestMixin):
    tester_class = GranitePluginTester
