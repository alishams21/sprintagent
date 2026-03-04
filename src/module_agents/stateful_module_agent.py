"""Stateful module planning agent using planner/designer/critic workflow.

This agent mirrors the multi-agent pattern used in project agents, but
operates on module-level plans (structure + tasks):

- Designer: creates and refines the module structure and task breakdown
- Critic: evaluates the module plan and suggests improvements with scored feedback
- Planner: orchestrates iterations between designer and critic via tools
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Runner, RunResult
from omegaconf import DictConfig

from src.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    AgentType,
    log_agent_usage,
)
from src.agent_utils.scoring import ModuleCritiqueWithScores
from src.module_agents.base_module_agent import BaseModuleAgent
from src.prompts import ModuleAgentPrompts, prompt_registry

console_logger = logging.getLogger(__name__)


class StatefulModuleAgent(BaseStatefulAgent, BaseModuleAgent):
    """Stateful module planning agent using planner/designer/critic workflow."""

    _uses_plan_checkpoints: bool = True

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize the module planning agent.

        Args:
            cfg: Hydra configuration for the agent.
            logger: Logger / run context with an ``output_dir`` attribute.
        """
        BaseModuleAgent.__init__(self, cfg=cfg, logger=logger)
        BaseStatefulAgent.__init__(self, cfg=cfg, logger=logger)

        # Per-run, per-agent session prefix (e.g. prompt_000_module_designer.db).
        output_dir_name = (
            Path(self.logger.output_dir).name
            if getattr(self.logger, "output_dir", None) is not None
            else ""
        )
        base = f"{output_dir_name}_" if output_dir_name else ""
        self.session_prefix: str = f"{base}module_"

        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=self.session_prefix
        )

        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None

        self.module_prompt: str = ""
        self.current_plan_text: str = ""

    # -------------------------------------------------------------------------
    # BaseStatefulAgent abstract interface implementations
    # -------------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        """Module planning agent type for the BaseStatefulAgent interface."""
        return AgentType.MODULE

    def _get_final_scores_directory(self) -> Path:
        """Directory where any final evaluation artifacts would be stored."""
        return Path(self.logger.output_dir) / "final_module_plan"

    def _get_critique_prompt_enum(self) -> Any:
        """Prompt enum for critic runner instruction."""
        return ModuleAgentPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_design_change_prompt_enum(self) -> Any:
        """Prompt enum for design-change instruction."""
        return ModuleAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    async def _get_critique_context_async(self) -> str:
        """Build critique context from the designer's session history."""
        session = getattr(self, "designer_session", None)
        if session is None:
            return ""

        try:
            items = await session.get_items()
        except Exception:
            return ""

        if not items:
            return ""

        texts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role not in ("user", "assistant"):
                continue
            content = item.get("content")

            if isinstance(content, str):
                texts.append(f"{role}: {content}")
            elif isinstance(content, list):
                part_texts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "input_text",
                        "text",
                    ):
                        text = part.get("text", "")
                        if text:
                            part_texts.append(text)
                if part_texts:
                    texts.append(f"{role}: {' '.join(part_texts)}")

        if not texts:
            return ""

        return "## Recent designer conversation (trimmed)\n\n" + "\n\n".join(texts)

    def _get_initial_design_prompt_enum(self) -> Any:
        """Prompt enum for initial design instruction."""
        return ModuleAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Template variables for initial design prompt (none for module agent)."""
        return {}

    # -------------------------------------------------------------------------
    # Plan checkpoint hooks (BaseStatefulAgent)
    # -------------------------------------------------------------------------

    def _on_designer_output(self, output: str) -> None:
        """Update current plan text after each designer output for checkpointing."""
        self.current_plan_text = output or ""

    def _get_plan_state_for_checkpoint(self) -> dict[str, Any] | None:
        """Return current plan state for checkpoint (N-1/N rollback)."""
        if not self.current_plan_text:
            return None
        return {"plan_text": self.current_plan_text}

    async def _apply_plan_checkpoint_rollback(
        self, plan_checkpoint: dict[str, Any]
    ) -> None:
        """Revert plan to previous checkpoint and notify designer session."""
        plan_text = plan_checkpoint.get("plan_text") or ""
        self.current_plan_text = plan_text
        rollback_msg = (
            "Resetting to previous plan version (scores regressed). "
            "The plan below is the reverted checkpoint."
        )
        try:
            if self.designer_session is not None:
                await self.designer_session.add_items(
                    [
                        {"role": "user", "content": rollback_msg},
                        {"role": "assistant", "content": plan_text},
                    ]
                )
        except Exception as e:
            console_logger.warning(
                "Failed to append plan rollback to designer session: %s", e
            )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _create_designer_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        """Create the module designer agent."""
        if tools is None:
            tools = []

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=ModuleAgentPrompts.DESIGNER_AGENT,
            module_prompt=self.module_prompt,
        )

    def _create_critic_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        """Create the module critic agent."""
        if tools is None:
            tools = []

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=ModuleAgentPrompts.CRITIC_AGENT,
            output_type=ModuleCritiqueWithScores,
            module_prompt=self.module_prompt,
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools using the shared BaseStatefulAgent implementation."""
        return super()._create_planner_tools()

    def _create_planner_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the planner agent that orchestrates designer and critic."""
        max_rounds = getattr(self.cfg, "max_critique_rounds", 3)
        early_finish_min_score = getattr(self.cfg, "early_finish_min_score", 8)

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=ModuleAgentPrompts.PLANNER_AGENT,
            module_prompt=self.module_prompt,
            max_critique_rounds=max_rounds,
            early_finish_min_score=early_finish_min_score,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def generate_module_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a module plan for the given prompt.

        This method:
        1. Initializes designer, critic, and planner agents
        2. Runs the planner with its runner instruction
        3. Saves the final plan to ``output_dir/module_plan.md`` and returns it
        """
        console_logger.info("Starting module planning workflow")

        self.module_prompt = prompt
        output_dir.mkdir(parents=True, exist_ok=True)

        self.designer = self._create_designer_agent()
        self.critic = self._create_critic_agent()
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(tools=planner_tools)

        runner_instruction = prompt_registry.get_prompt(
            prompt_enum=ModuleAgentPrompts.PLANNER_RUNNER_INSTRUCTION,
        )

        planner_max_turns = getattr(
            getattr(self.cfg.agents, "planner_agent", None), "max_turns", 64
        )
        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=planner_max_turns,
        )

        log_agent_usage(result=result, agent_name="PLANNER (MODULE)")

        final_plan = result.final_output or ""

        plan_path = output_dir / "module_plan.md"
        plan_path.write_text(final_plan, encoding="utf-8")
        console_logger.info(f"Module plan saved to: {plan_path}")

        return final_plan
