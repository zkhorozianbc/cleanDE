from datetime import datetime

import pytest


@pytest.fixture()
def high_date() -> datetime:
    return datetime(9999, 12, 31)


@pytest.fixture()
def t0() -> datetime:
    return datetime(2024, 1, 1)


@pytest.fixture()
def t1() -> datetime:
    return datetime(2024, 6, 1)
