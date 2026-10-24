from pathlib import Path

import pytest

from src.config import Settings
from src.datasets import DatasetStore, parse_csv


@pytest.fixture
def sample_bytes():
    return (Path(__file__).resolve().parents[1] / "data/tickets_sample.csv").read_bytes()


@pytest.fixture
def settings(tmp_path):
    return Settings(runtime_dir=tmp_path, retry_delay=0)


@pytest.fixture
def frame(sample_bytes, settings):
    return parse_csv(sample_bytes, settings)


@pytest.fixture
def registered(sample_bytes, settings):
    store = DatasetStore(settings)
    return store, store.register(sample_bytes)
