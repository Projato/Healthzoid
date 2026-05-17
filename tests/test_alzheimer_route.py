import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeImage:
    def reshape(self, *_args, **_kwargs):
        return self

    def __truediv__(self, _other):
        return self


class FakePredictionRow:
    def __init__(self, prediction_class):
        self.prediction_class = prediction_class

    def argmax(self):
        return self.prediction_class


class FakeModel:
    def __init__(self, prediction_class=0):
        self.prediction_class = prediction_class

    def predict(self, _img):
        return [FakePredictionRow(self.prediction_class)]


class FakeFlask:
    def __init__(self, _name):
        self.config = {}
        self.secret_key = None

    def route(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def after_request(self, func):
        return func


class FakeUpload:
    filename = "scan.jpg"

    def save(self, _path):
        return None


@pytest.fixture()
def app_module(monkeypatch):
    fake_flask = types.ModuleType("flask")
    fake_flask.Flask = FakeFlask
    fake_flask.flash = lambda *_args, **_kwargs: None
    fake_flask.redirect = lambda location: location
    fake_flask.url_for = lambda endpoint, **_kwargs: endpoint
    fake_flask.request = types.SimpleNamespace(method="GET", form={}, files={}, url="/")
    fake_flask.render_template = lambda template, **context: context
    monkeypatch.setitem(sys.modules, "flask", fake_flask)

    fake_werkzeug = types.ModuleType("werkzeug")
    fake_werkzeug_utils = types.ModuleType("werkzeug.utils")
    fake_werkzeug_utils.secure_filename = lambda filename: filename
    monkeypatch.setitem(sys.modules, "werkzeug", fake_werkzeug)
    monkeypatch.setitem(sys.modules, "werkzeug.utils", fake_werkzeug_utils)

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imread = lambda _path: FakeImage()
    fake_cv2.resize = lambda image, _size: image
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    fake_imutils = types.ModuleType("imutils")
    fake_imutils.grab_contours = lambda contours: contours
    monkeypatch.setitem(sys.modules, "imutils", fake_imutils)

    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda value: value
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    fake_tensorflow = types.ModuleType("tensorflow")
    fake_keras = types.ModuleType("tensorflow.keras")
    fake_models = types.ModuleType("tensorflow.keras.models")
    fake_models.load_model = lambda _path: FakeModel()
    fake_applications = types.ModuleType("tensorflow.keras.applications")
    fake_vgg16 = types.ModuleType("tensorflow.keras.applications.vgg16")
    fake_vgg16.preprocess_input = lambda image: image
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", fake_keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", fake_models)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications", fake_applications)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications.vgg16", fake_vgg16)

    fake_joblib = types.ModuleType("joblib")
    fake_joblib.load = lambda _path: object()
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    import pickle

    monkeypatch.setattr(pickle, "load", lambda _file: object())

    sys.modules.pop("app", None)
    return importlib.import_module("app")


def _post_alzheimer_result(app_module, prediction_class):
    app_module.alzheimer_model = FakeModel(prediction_class)
    app_module.request = types.SimpleNamespace(
        method="POST",
        url="/resulta",
        form={
            "firstname": "Ada",
            "lastname": "Lovelace",
            "email": "ada@example.com",
            "phone": "5555555555",
            "gender": "Female",
            "age": "42",
        },
        files={"file": FakeUpload()},
    )

    def render_result(template, **context):
        assert template == "resulta.html"
        return f"Result:<i> {context['alzheimer_classes'][context['r']]} </i>"

    app_module.render_template = render_result
    return app_module.resulta()


def test_alzheimer_route_renders_different_mocked_prediction_labels(app_module):
    first_result = _post_alzheimer_result(app_module, 0)
    second_result = _post_alzheimer_result(app_module, 2)

    assert "NonDemented" in first_result
    assert "MildDemented" in second_result
    assert first_result != second_result


def test_alzheimer_template_classes_match_training_labels(app_module):
    notebook = json.loads((ROOT / "Alzheimer Detection/Alzheimer Detection.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "CLASSES" in "".join(cell.get("source", []))
    )
    training_labels = re.findall(r"'([^']*Demented)'", source)[:4]

    template = (ROOT / "templates/resulta.html").read_text()

    assert app_module.ALZHEIMER_CLASSES == training_labels
    assert "alzheimer_classes[r]" in template
