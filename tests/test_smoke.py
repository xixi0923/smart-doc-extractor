"""Minimal smoke test for CI.

Ensures the project's root ``config`` module imports without error, guarding
against import-time regressions. Expand with real unit tests over time.
"""


def test_config_importable():
    import config  # noqa: F401

    assert config is not None
