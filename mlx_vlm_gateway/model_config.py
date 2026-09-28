"""Resolve a model's installed worker without exposing engine choices to Junie."""

from pathlib import Path


def select_model(config: dict, model: str, descriptor: dict) -> dict:
    selected = {**config, "model_name": model}
    backend = descriptor.get("worker_backend", "mlx")
    if backend not in {"mlx", "splash"}:
        raise ValueError(f"Unsupported model backend: {backend}")
    selected["worker_backend"] = backend
    if "draft_model" in descriptor:
        selected["draft_model"] = descriptor["draft_model"]
    elif backend != config.get("worker_backend", "mlx"):
        raise ValueError("Switching engines requires an explicit draft_model")
    if descriptor.get("draft_kind"):
        selected["draft_kind"] = descriptor["draft_kind"]
    if backend == "splash":
        name = descriptor.get("splash_package")
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or Path(name).name != name
        ):
            raise ValueError("Invalid Splash package name in model descriptor")
        directory = Path(selected["models_dir"]).expanduser().resolve()
        package = (directory / name).resolve()
        if not package.is_relative_to(directory):
            raise ValueError("Splash package is outside the managed models directory")
        selected["splash_package"] = str(package)
        from .splash import validate_config

        validate_config(selected)
    else:
        selected.pop("splash_package", None)
    return selected
