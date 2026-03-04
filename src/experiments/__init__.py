"""Plan-generation experiments (project / module / task)."""

from omegaconf import DictConfig

from .base_experiment import BaseExperiment
from .plan_generation import PlanGenerationExperiment

# Keys must match yaml filenames under configurations/experiment/ (without .yaml).
exp_registry = {
    "plan_project": PlanGenerationExperiment,
}


def build_experiment(cfg: DictConfig) -> BaseExperiment:
    """Build an experiment from config.

    Args:
        cfg: Full config including experiment group.

    Returns:
        Experiment instance.

    Raises:
        ValueError: If cfg.experiment._name is not in the registry.
    """
    name = cfg.experiment._name
    if name not in exp_registry:
        raise ValueError(
            f"Experiment {name!r} not found. "
            f"Registered: {list(exp_registry.keys())}. "
            "Add the experiment class to exp_registry in src/experiments/__init__.py "
            "with the same key as the yaml file under configurations/experiment/."
        )
    return exp_registry[name](cfg)
