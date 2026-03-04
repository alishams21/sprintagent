"""Base class for plan-generation experiments (project / module / task)."""

from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.module_agents.base_module_agent import BaseModuleAgent
from src.project_agents.base_project_agent import BaseProjectAgent
from src.task_agents.base_task_agent import BaseTaskAgent
from src.utils.logging import BaseLogger
from src.visualization_agents.base_visualization_agent import BaseVisualizationAgent


class BaseExperiment(ABC):
    """
    Abstract base class for plan-generation experiments.

    Experiments define tasks (e.g. plan_project) that run sequentially.
    Each experiment specifies compatible project/module/task agents and supports
    parallel or single runs over multiple prompts.
    """

    # Each key must match a yaml file under configurations/<agent_type>_agent/<key>.yaml
    compatible_project_agents: dict[str, type] = {}
    compatible_module_agents: dict[str, type] = {}
    compatible_task_agents: dict[str, type] = {}
    compatible_visualization_agents: dict[str, type] = {}

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.experiment.output_dir)

    @staticmethod
    def build_project_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
    ) -> BaseProjectAgent:
        """Build project agent from config.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig.
            compatible_agents: Mapping of agent config name to agent class.
            logger: Logger with output_dir for the agent.

        Returns:
            Project agent instance.
        """
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )
        agent_config = config_dict["project_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Project agent '{agent_name}' not found in compatible_project_agents. "
                "Ensure each key matches a yaml file under configurations/project_agent "
                "without the .yaml suffix."
            )

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
        )

    @staticmethod
    def build_module_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
    ) -> BaseModuleAgent:
        """Build module agent from config."""
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )
        agent_config = config_dict["module_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Module agent '{agent_name}' not found in compatible_module_agents."
            )

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
        )

    @staticmethod
    def build_task_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
    ) -> BaseTaskAgent:
        """Build task agent from config."""
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )
        agent_config = config_dict["task_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Task agent '{agent_name}' not found in compatible_task_agents."
            )

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
        )

    @staticmethod
    def build_visualization_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
    ) -> BaseVisualizationAgent:
        """Build visualization agent from config."""
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )
        agent_config = config_dict.get("visualization_agent")
        if agent_config is None:
            raise ValueError(
                "Config has no 'visualization_agent' key. "
                "Add visualization_agent to defaults in config.yaml."
            )
        agent_name = agent_config.get("_name")
        if not agent_name:
            raise ValueError("visualization_agent._name is required.")
        if agent_name not in compatible_agents:
            raise ValueError(
                f"Visualization agent '{agent_name}' not found in "
                "compatible_visualization_agents."
            )
        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
        )

    def exec_task(self, task_name: str) -> None:
        """Execute a task by name."""
        if hasattr(self, task_name):
            getattr(self, task_name)()
        else:
            raise ValueError(
                f"Task '{task_name}' not found in experiment '{self.__class__.__name__}'"
            )

    @abstractmethod
    def plan_project(self) -> None:
        """Run project planning (and optionally module/task stages) for configured prompts."""
