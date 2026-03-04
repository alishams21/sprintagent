"""Stateful project planning agent using planner/designer/critic workflow.

This agent mirrors the multi-agent pattern used in scene-generation agents, but
operates purely on textual project plans:

- Designer: creates and refines the hierarchical plan (project → modules → tasks)
- Critic: evaluates the plan and suggests improvements with scored feedback
- Planner: orchestrates iterations between designer and critic via tools
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Runner, RunResult
from omegaconf import DictConfig

from src.agent_utils.base_stateful_agent import BaseStatefulAgent, AgentType, log_agent_usage
from src.agent_utils.scoring import ProjectCritiqueWithScores
from src.project_agents.base_project_agent import BaseProjectAgent
from src.prompts import ProjectAgentPrompts, prompt_registry

console_logger = logging.getLogger(__name__)


class StatefulProjectAgent(BaseStatefulAgent, BaseProjectAgent):
    """Stateful project planning agent using planner/designer/critic workflow."""

    _uses_plan_checkpoints: bool = True

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize the project planning agent.

        Args:
            cfg: Hydra configuration for the agent.
            logger: Logger / run context with an ``output_dir`` attribute.
        """
        # Initialize both the project-specific base and the shared stateful base.
        BaseProjectAgent.__init__(self, cfg=cfg, logger=logger)
        BaseStatefulAgent.__init__(self, cfg=cfg, logger=logger)

        # Per-run, per-agent session prefix so each agent's designer/critic
        # sessions are isolated (e.g. prompt_000_project_designer.db).
        output_dir_name = Path(self.logger.output_dir).name if getattr(
            self.logger, "output_dir", None
        ) is not None else ""
        base = f"{output_dir_name}_" if output_dir_name else ""
        self.session_prefix: str = f"{base}project_"

        # Persistent agent sessions (reuse BaseStatefulAgent implementation).
        # Both designer and critic sessions share the same prefix so they are
        # tied to the same logical run.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=self.session_prefix
        )

        # Agent instances (created lazily when running a plan).
        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None

        # Project prompt describing the overall plan (set when generating a plan).
        self.project_prompt: str = ""
        # Current plan text (updated after each designer output; used for checkpoints).
        self.current_plan_text: str = ""

    # -------------------------------------------------------------------------
    # BaseStatefulAgent abstract interface implementations
    # -------------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        """Project planning agent type for the BaseStatefulAgent interface."""
        return AgentType.PROJECT

    def _get_final_scores_directory(self) -> Path:
        """Directory where any final evaluation artifacts would be stored."""
        return Path(self.logger.output_dir) / "final_project_plan"

    def _get_critique_prompt_enum(self) -> Any:
        """Prompt enum for critic runner instruction (unused by current tools)."""
        return ProjectAgentPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_design_change_prompt_enum(self) -> Any:
        """Prompt enum for design-change instruction (unused by current tools)."""
        return ProjectAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    async def _get_critique_context_async(self) -> str:
        """Build critique context from the designer's session history.

        This leverages TurnTrimmingSession (when enabled) so that older turns are
        summarized and only the most relevant recent context is expanded.
        """
        session = getattr(self, "designer_session", None)
        if session is None:
            return ""

        try:
            items = await session.get_items()
        except Exception:
            # Fail open: if session access fails, return no extra context.
            return ""

        if not items:
            return ""

        # Collect recent user + assistant messages (which may already include
        # summaries from TurnTrimmingSession for older turns).
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
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                        text = part.get("text", "")
                        if text:
                            part_texts.append(text)
                if part_texts:
                    texts.append(f"{role}: {' '.join(part_texts)}")

        if not texts:
            return ""

        return "## Recent designer conversation (trimmed)\n\n" + "\n\n".join(texts)

    def _get_initial_design_prompt_enum(self) -> Any:
        """Prompt enum for initial design instruction (unused by current tools)."""
        return ProjectAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Template variables for initial design prompt (none for project agent)."""
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
        """Create the project designer agent.

        Mirrors the pattern used in other stateful agents by delegating to
        BaseStatefulAgent._create_designer_agent, while keeping the existing
        project-specific prompt configuration.
        """
        if tools is None:
            tools = []

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.DESIGNER_AGENT,
            project_prompt=self.project_prompt,
        )

    def _create_critic_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        """Create the project critic agent.

        Mirrors the pattern used in other stateful agents by delegating to
        BaseStatefulAgent._create_critic_agent, while keeping the existing
        project-specific prompt configuration.
        """
        if tools is None:
            tools = []

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.CRITIC_AGENT,
            output_type=ProjectCritiqueWithScores,
            project_prompt=self.project_prompt,
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools using the shared BaseStatefulAgent implementation."""
        return super()._create_planner_tools()

    def _create_planner_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the planner agent that orchestrates designer and critic.

        Delegates to BaseStatefulAgent._create_planner_agent so the planner
        shares the same configuration pattern (model settings, etc.) as other
        stateful agents while using project-specific prompts.
        """
        max_rounds = getattr(self.cfg, "max_critique_rounds", 3)
        early_finish_min_score = getattr(self.cfg, "early_finish_min_score", 8)

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.PLANNER_AGENT,
            project_prompt=self.project_prompt,
            max_critique_rounds=max_rounds,
            early_finish_min_score=early_finish_min_score,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def generate_project_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a hierarchical project plan for the given prompt.

        This method:
        1. Initializes designer, critic, and planner agents
        2. Runs the planner with its runner instruction
        3. Saves the final plan to ``output_dir`` and returns it
        """
        console_logger.info("Starting project planning workflow")

        self.project_prompt = prompt
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create agents.
        self.designer = self._create_designer_agent()
        self.critic = self._create_critic_agent()
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(tools=planner_tools)

        # Get runner instruction.
        runner_instruction = prompt_registry.get_prompt(
            prompt_enum=ProjectAgentPrompts.PLANNER_RUNNER_INSTRUCTION,
        )

        # Run planning workflow.
        planner_max_turns = getattr(
            getattr(self.cfg.agents, "planner_agent", None), "max_turns", 64
        )
        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=planner_max_turns,
        )

        log_agent_usage(result=result, agent_name="PLANNER (PROJECT)")

        final_plan = result.final_output or ""

        # Save final plan to disk.
        plan_path = output_dir / "project_plan.md"
        plan_path.write_text(final_plan, encoding="utf-8")
        console_logger.info(f"Project plan saved to: {plan_path}")

        return final_plan

