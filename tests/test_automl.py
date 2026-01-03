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
        "No AutoML module found. Expected one of: " + ", ".join(module_names)
    )


def _get_first_candidate(module, names: list[str]):
    for name in names:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    pytest.skip("No AutoML entrypoint found. Expected one of: " + ", ".join(names))


def _init_if_possible(cls):
    signature = inspect.signature(cls)
    required = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        pytest.skip("AutoML class requires init args without defaults; provide defaults.")
    return cls()


def _fit_if_available(model, features, target):
    fit = getattr(model, "fit", None)
    if callable(fit):
        fit(features, target)


def _predict_if_available(model, features):
    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict(features)
    pytest.skip("AutoML model lacks predict method.")


def _call_model_builder(builder, features, target):
    signature = inspect.signature(builder)
    params = signature.parameters
    kwargs = {}
    if "X" in params:
        kwargs["X"] = features
    if "y" in params:
        kwargs["y"] = target
    if kwargs:
        return builder(**kwargs)
    required = [
        param
        for param in params.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        pytest.skip("AutoML builder requires args without defaults; provide defaults.")
    return builder()


def test_automl_model_predicts():
    module = _import_first(["automl", "auto_ml", "src.automl", "src.auto_ml"])
    candidate = _get_first_candidate(
        module,
        ["AutoML", "AutoMLModel", "build_model", "train_model", "fit_model"],
    )
    features = [[0], [1], [2]]
    target = [0, 1, 1]
    if inspect.isclass(candidate):
        model = _init_if_possible(candidate)
    else:
        model = _call_model_builder(candidate, features, target)

    _fit_if_available(model, features, target)
    predictions = _predict_if_available(model, features)
    assert predictions is not None
    assert len(predictions) == len(features)
