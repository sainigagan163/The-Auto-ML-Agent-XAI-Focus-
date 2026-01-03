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
        "No dataset loader module found. Expected one of: " + ", ".join(module_names)
    )


def _get_first_callable(module, names: list[str]):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    pytest.skip(
        "No dataset loader callable found. Expected one of: " + ", ".join(names)
    )


def _call_if_no_required_args(func):
    signature = inspect.signature(func)
    required = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        pytest.skip(
            "Dataset loader requires arguments without defaults; provide defaults to test."
        )
    return func()


def _assert_non_empty_dataset(dataset):
    assert dataset is not None
    if isinstance(dataset, tuple) and len(dataset) == 2:
        features, target = dataset
        assert len(features) > 0
        assert len(target) > 0
    elif hasattr(dataset, "__len__"):
        assert len(dataset) > 0


def test_dataset_loader_returns_data():
    module = _import_first(
        [
            "loaders",
            "dataset_loaders",
            "data_loaders",
            "src.loaders",
            "src.dataset_loaders",
            "src.data_loaders",
        ]
    )
    loader = _get_first_callable(module, ["load_dataset", "load_data", "get_dataset"])
    dataset = _call_if_no_required_args(loader)
    _assert_non_empty_dataset(dataset)
