import importlib
import inspect
import pytest


def _import_first(module_names: list[str]):
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    pytest.skip(
        "No feature engineering module found. Expected one of: " + ", ".join(module_names)
    )


def _get_first_callable(module, names: list[str]):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    pytest.skip(
        "No feature engineering callable found. Expected one of: " + ", ".join(names)
    )


def _call_with_sample_input(func, sample):
    signature = inspect.signature(func)
    params = signature.parameters
    required = [
        param
        for param in params.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) == 0:
        return func()
    if len(required) == 1:
        return func(sample)
    pytest.skip(
        "Feature engineering requires multiple arguments without defaults; provide defaults to test."
    )


def _get_shape(data):
    if hasattr(data, "shape"):
        shape = data.shape
        if len(shape) >= 2:
            return shape[0], shape[1]
    if hasattr(data, "__len__") and len(data) > 0:
        first = data[0]
        if hasattr(first, "__len__"):
            return len(data), len(first)
    return None


def test_feature_engineering_output_shape():
    module = _import_first(
        [
            "feature_engineering",
            "features",
            "src.feature_engineering",
            "src.features",
        ]
    )
    engineer = _get_first_callable(
        module,
        ["build_features", "make_features", "transform_features", "engineer_features"],
    )
    sample_input = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    features = _call_with_sample_input(engineer, sample_input)
    shape = _get_shape(features)
    assert shape is not None
    rows, cols = shape
    assert rows == len(sample_input)
    assert cols > 0
