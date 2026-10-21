import pytest
from unittest.mock import patch
import numpy as np
from app import app as flask_app

@pytest.fixture
def app():
    flask_app.config.update({"TESTING": True})
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture(autouse=True)
def mock_model_functions(request):
    if "test_model.py" in request.node.fspath.strpath:
        yield
    else:
        with patch("app.preprocess_img", return_value=np.zeros((1, 28, 28, 1))):
            with patch("app.predict_result", return_value=7):
                yield
