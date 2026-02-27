import pytest

def pytest_addoption(parser):
    opts = [
        ('--engine-dir', dict(default=None, help='Engine directory')),
        ('--trtf-binary', dict(default=None, help='Path to trtf binary')),
        ('--hf-python', dict(default=None, help='Python with HF tokenizers')),
        ('--rebuild-engines', dict(action='store_true', default=False, help='Rebuild bundles')),
        ('--e2e-task-strategy', dict(default=None, help='Filter by task strategy')),
        ('--e2e-artifacts-dir', dict(default=None, help='Artifacts output dir')),
    ]
    for name, kw in opts:
        try:
            parser.addoption(name, **kw)
        except ValueError:
            pass
