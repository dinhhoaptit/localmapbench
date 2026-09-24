"""Smoke test for package import."""

import localmapbench


def test_version():
    assert localmapbench.__version__ == "0.2.0"
