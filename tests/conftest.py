import pytest

from ocear import io


@pytest.fixture
def img():
    return io.load_image('tests/fixtures/example.jpg')
