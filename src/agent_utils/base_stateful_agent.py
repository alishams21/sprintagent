"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for plan-based agents (e.g. project,
module, task planning). It does not assume scene generation: scene/rendering
and checkpoint rollback are optional and only used when _uses_scene_checkpoints
is True. Agents like StatefulProjectAgent use only sessions, designer/critic/
planner creation, and design-change flows.
"""

import copy
import logging
import shutil

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    RunResult,
    SQLiteSession,
    function_tool,
)
from agents.memory.session import Session
from omegaconf import DictConfig
from openai import Timeout
from openai.types.shared import Reasoning

from src.agent_utils.checkpoint_state import (
    initialize_checkpoint_attributes,
    initialize_plan_checkpoint_attributes,
)
from src.agent_utils.intra_turn_image_filter import IntraTurnImageFilter
from src.agent_utils.scoring import (
    CritiqueWithScores,
    compute_total_score,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from src.agent_utils.turn_trimming_session import TurnTrimmingSession
from src.prompts import prompt_registry
from src.utils.logging import BaseLogger
from src.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent type for plan-based workflows (project/module/task/visualization)."""

    PROJECT = "project"
    MODULE = "module"
    TASK = "task"
    VISUALIZATION = "visualization"


def log_agent_usage(result: RunResult, agent_name: str) -> None:
    """Log token usage from an agent run.

    Args:
        result: The RunResult from Runner.run().
        agent_name: Human-readable name for the agent (e.g., "DESIGNER", "CRITIC").
    """
    usage = result.context_wrapper.usage
    cached = (
        usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    )
    reasoning = (
        usage.output_tokens_details.reasoning_tokens
        if usage.output_tokens_details
        else 0
    )
    # Get final context size from last request (context only grows during a run).
    final_context = (
        usage.request_usage_entries[-1].input_tokens
        if usage.request_usage_entries
        else usage.input_tokens
    )
    console_logger.info(
        f"[{agent_name}] Token usage: "
        f"input={usage.input_tokens:,}, "
        f"output={usage.output_tokens:,}, "
        f"reasoning={reasoning:,}, "
        f"cached={cached:,}, "
        f"total={usage.total_tokens:,}, "
        f"requests={usage.requests}, "
        f"final_context_length={final_context:,}"
    )


class BaseStatefulAgent(ABC):
    """Base class for stateful agents with planner/designer/critic workflow.

    Provides shared infrastructure for plan-based agents (project, module, task):
    - Session management (SQLiteSession for designer/critic)
    - Agent creation (designer, critic, planner) with domain-specific prompts
    - Optional checkpoint/rollback and placement style when _uses_scene_checkpoints
      or _is_placement_agent are True (e.g. for scene-generation agents).

    Subclasses implement abstract methods for prompts and (optionally) placement
    noise. No scene or rendering is required unless the subclass uses them.
    """

    # Set True only for agents that have scene state and use checkpoint rollback.
    _uses_scene_checkpoints: bool = False
    # Set True for plan-based agents (e.g. project) that checkpoint plan state for rollback.
    _uses_plan_checkpoints: bool = False
    # Set to a tool name (e.g. "observe_scene") to force critic's first tool call; None for plan-only agents.
    _critic_tool_choice: str | None = None

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent for collision filtering.

        Each agent type can only modify certain object types:
        - PROJECT: Project plan
        - MODULE: Module plan
        - TASK: Task plan

        Returns:
            AgentType for this agent.
        """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
    ):
        """Initialize base placement agent with shared infrastructure.

        Args:
            cfg: Hydra configuration object.
            logger: Logger for experiment tracking.

        """
        self.cfg = cfg
        self.logger = logger

        # Use global prompt registry (same pattern as domain base classes).
        self.prompt_registry = prompt_registry

        # Initialize checkpoint state (N-1 and N pattern for rollback).
        initialize_checkpoint_attributes(target=self)
        initialize_plan_checkpoint_attributes(target=self)


    def _get_model_settings(
        self,
        settings_key: str | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ModelSettings | None:
        """Create ModelSettings with timeout, reasoning effort, verbosity, and tool.

        Args:
            settings_key: Key in cfg.openai.reasoning_effort and cfg.openai.verbosity
                for this agent (e.g., "designer", "critic", "planner"). If None,
                no reasoning effort or verbosity is set.
            tool_choice: Tool name to force as first call (e.g., "observe_scene").
                Resets after first tool call by default to prevent infinite loops.
            parallel_tool_calls: Whether to allow parallel tool calls. Set to False
                for planner agents to prevent race conditions on shared sessions.

        Returns:
            ModelSettings with timeout, reasoning, verbosity, and tool_choice if
            configured, None otherwise.
        """
        kwargs: dict = {}
        extra_args: dict = {}

        # Add timeout if configured (api_timeout is optional).
        if hasattr(self.cfg, "api_timeout"):
            timeout_cfg = self.cfg.api_timeout
            timeout = Timeout(
                connect=timeout_cfg.connect,
                read=timeout_cfg.read,
                write=timeout_cfg.write,
                pool=timeout_cfg.pool,
            )
            extra_args["timeout"] = timeout

        # Add service_tier if configured (non-null/non-empty).
        service_tier = getattr(self.cfg.openai, "service_tier", None)
        if service_tier:
            extra_args["service_tier"] = service_tier

        if extra_args:
            kwargs["extra_args"] = extra_args

        # Add reasoning effort and verbosity if key is provided.
        if settings_key:
            reasoning_cfg = getattr(self.cfg.openai, "reasoning_effort", None)
            if reasoning_cfg is not None:
                effort = getattr(reasoning_cfg, settings_key, None)
                if effort is not None and str(effort).lower() != "none":
                    kwargs["reasoning"] = Reasoning(effort=effort)

            verbosity_cfg = self.cfg.openai.verbosity
            verbosity = getattr(verbosity_cfg, settings_key)
            kwargs["verbosity"] = verbosity

        # Add tool_choice to force specific tool call first.
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Add parallel_tool_calls setting if specified.
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        return ModelSettings(**kwargs) if kwargs else None

    def _create_designer_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create designer agent with tools and domain-specific prompt.

        This method provides the shared pattern for designer agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the designer.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        return Agent(
            name=designer_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            model_settings=self._get_model_settings(settings_key="designer"),
        )

    def _create_critic_agent(
        self,
        tools: list[FunctionTool],
        prompt_enum: Any,
        output_type: type[CritiqueWithScores],
        **prompt_kwargs: Any,
    ) -> Agent:
        """Create critic agent with structured output.

        This method provides the shared pattern for critic agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the critic.
            prompt_enum: Prompt enum from domain-specific registry.
            output_type: CritiqueWithScores subclass for structured output.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured critic agent with domain-specific CritiqueWithScores type.
        """
        critic_config = self.cfg.agents.critic_agent
        return Agent(
            name=critic_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            output_type=output_type,
            model_settings=self._get_model_settings(
                settings_key="critic",
                tool_choice=getattr(self, "_critic_tool_choice", None),
            ),
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create planner agent for workflow coordination.

        This method provides the shared pattern for planner agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the planner.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        return Agent(
            name=planner_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            # Disable parallel tool calls to prevent race conditions on shared
            # sessions (designer_session, critic_session). When the model returns
            # multiple tool calls in one response, they would otherwise run
            # concurrently and cause SQLite locking issues.
            model_settings=self._get_model_settings(
                settings_key="planner", parallel_tool_calls=False
            ),
        )

    def _create_sessions(self, session_prefix: str = "") -> tuple[Session, Session]:
        """Create designer and critic sessions for persistent conversation history.

        Sessions are optionally wrapped with TurnTrimmingSession for memory
        management if session_memory is enabled in config.

        Args:
            session_prefix: Optional prefix for session IDs (e.g., furniture ID).

        Returns:
            Tuple of (designer_session, critic_session).
        """
        designer_id = f"{session_prefix}designer" if session_prefix else "designer"
        critic_id = f"{session_prefix}critic" if session_prefix else "critic"

        designer_sqlite = SQLiteSession(
            session_id=designer_id,
            db_path=self.logger.output_dir / f"{designer_id}.db",
        )
        critic_sqlite = SQLiteSession(
            session_id=critic_id,
            db_path=self.logger.output_dir / f"{critic_id}.db",
        )

        # Wrap with memory management if configured.
        memory_cfg = getattr(self.cfg, "session_memory", None)
        if memory_cfg and getattr(memory_cfg, "enabled", False):
            console_logger.info(
                f"Enabling turn-trimming session (keep_last_n_turns="
                f"{memory_cfg.keep_last_n_turns}, summarization="
                f"{memory_cfg.enable_summarization})"
            )
            designer_session: Session = TurnTrimmingSession(
                wrapped_session=designer_sqlite, cfg=self.cfg
            )
            critic_session: Session = TurnTrimmingSession(
                wrapped_session=critic_sqlite, cfg=self.cfg
            )
        else:
            designer_session = designer_sqlite
            critic_session = critic_sqlite

        return designer_session, critic_session

    def _create_run_config(self) -> RunConfig:
        """Create RunConfig with optional intra-turn image filter.

        When session_memory.intra_turn_observation_stripping is enabled (e.g.
        for scene agents that use observe_scene), adds a filter to reduce
        token usage. Plan-only agents typically leave this disabled.

        Returns:
            RunConfig with optional call_model_input_filter, or empty.
        """
        session_memory = getattr(self.cfg, "session_memory", None)
        if session_memory is None:
            return RunConfig()
        intra_cfg = getattr(session_memory, "intra_turn_observation_stripping", None)
        if intra_cfg is not None and getattr(intra_cfg, "enabled", False):
            return RunConfig(call_model_input_filter=IntraTurnImageFilter(cfg=self.cfg))
        return RunConfig()

    @abstractmethod
    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final scores/artifacts.

        Returns:
            Path to the directory where final scores or outputs should be saved.
        """

    async def _finalize_scene_and_scores(self) -> None:
        """Optionally copy scores/renders to final directory.

        No-op for plan-only agents (no scene/rendering). Subclasses that
        use scores/renders can override to copy them to _get_final_scores_directory().
        """
        render_dir = getattr(self, "final_render_dir", None) or getattr(
            self, "checkpoint_render_dir", None
        )
        if render_dir is None:
            return
        final_dir = self._get_final_scores_directory()
        final_dir.mkdir(parents=True, exist_ok=True)
        scores_source = Path(render_dir) / "scores.yaml"
        if scores_source.exists():
            shutil.copy(scores_source, final_dir / "scores.yaml")
            console_logger.info(f"Saved final scores to {final_dir / 'scores.yaml'}")
        render_images = list(Path(render_dir).glob("*.png"))
        for img_path in render_images:
            shutil.copy(img_path, final_dir / img_path.name)
        if render_images:
            console_logger.info(
                f"Copied {len(render_images)} render images to {final_dir}"
            )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools for the design workflow.

        Returns tools that the planner uses to coordinate designer and critic:
        - request_initial_design: Request initial design from designer
        - request_critique: Request evaluation from critic
        - request_design_change: Request design modifications based on feedback

        Returns:
            List of function tools for planner agent.
        """

        @function_tool
        async def request_initial_design() -> str:
            """Request the designer to create the initial design.

            The designer will analyze the context and create an appropriate
            initial layout or arrangement.

            Returns:
                Designer's report of what was created and why.
            """
            return await self._request_initial_design_impl()

        @function_tool
        async def request_critique(is_final_round: bool = False) -> str:
            """Request the critic to evaluate the current design.

            The critic will examine the current state and provide feedback
            on what works well and what needs improvement.

            Args:
                is_final_round: Set to True when this is the last critique before
                    completion (e.g. stop condition met or max rounds reached).
                    When True, checkpoint state is not updated so the previous
                    iteration remains available for rollback comparison.

            Returns:
                Critic's detailed evaluation with specific improvement suggestions.
            """
            return await self._request_critique_impl(update_checkpoint=not is_final_round)

        @function_tool
        async def request_design_change(instruction: str) -> str:
            """Request the designer to address specific issues.

            Based on the critic's feedback, provide clear instructions about
            what to change. The designer will modify the design to address
            the issues while maintaining what works well.

            Args:
                instruction: Specific changes to make based on critique feedback.

            Returns:
                Designer's report of what was changed.
            """
            return await self._request_design_change_impl(instruction)

        @function_tool
        async def reset_plan_to_previous_checkpoint() -> str:
            """Revert the plan to the previous checkpoint if the latest scores regressed.

            Compare the latest critique scores with the previous checkpoint scores.
            If the total score decreased, reset the plan to the last saved version
            (N-1) so the designer can continue from a better state. Use when
            iteration made things worse.

            Returns:
                Message describing whether a reset was performed and the outcome.
            """
            return await self._perform_plan_checkpoint_reset()

        tools: list[FunctionTool] = [request_initial_design]

        # Only add critique-related tools if critique rounds are enabled.
        if self.cfg.max_critique_rounds > 0:
            tools.extend([request_critique, request_design_change])
            if getattr(self, "_uses_plan_checkpoints", False):
                tools.append(reset_plan_to_previous_checkpoint)

        return tools

    @abstractmethod
    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Prompt enum for domain-specific critic instruction.
        """

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra keyword arguments for critic prompt template.

        Override in subclasses to inject domain-specific context into critic prompts.
        For example, furniture agent overrides this to add reachability context.

        Returns:
            Dictionary of extra kwargs to pass to prompt rendering.
        """
        return {}

    def _get_critique_context(self) -> str:
        """Extra context appended to the critique instruction (e.g. current plan).

        Override in subclasses so the critic sees what to evaluate. Default: none.

        Returns:
            String to append after the critique instruction, or empty.
        """
        return ""

    def _on_designer_output(self, output: str) -> None:
        """Called after the designer produces output (initial design or design change).

        Override in subclasses to update derived state (e.g. current_plan_text
        for plan checkpointing). Default: no-op.

        Args:
            output: The designer's final output text.
        """
        pass

    def _get_plan_state_for_checkpoint(self) -> dict[str, Any] | None:
        """Return current plan state for checkpointing (plan-based agents only).

        Override in subclasses that set _uses_plan_checkpoints. Return a dict
        (e.g. {"plan_text": self.current_plan_text}) to checkpoint; return None
        to skip checkpoint update.

        Returns:
            Dict suitable for copy.deepcopy and rollback, or None.
        """
        return None

    async def _apply_plan_checkpoint_rollback(
        self, plan_checkpoint: dict[str, Any]
    ) -> None:
        """Apply rollback to a previous plan checkpoint (plan-based agents only).

        Override in subclasses that set _uses_plan_checkpoints. Typically:
        set current_plan_text from plan_checkpoint["plan_text"] and optionally
        append a message to the designer session summarizing the rollback.

        Args:
            plan_checkpoint: The N-1 checkpoint dict (e.g. {"plan_text": "..."}).
        """
        pass

    async def _perform_plan_checkpoint_reset(self) -> str:
        """Compare current vs previous checkpoint scores and rollback if regressed.

        Used by the reset_plan_to_previous_checkpoint planner tool. If total
        score decreased, applies _apply_plan_checkpoint_rollback(previous_plan_checkpoint).

        Returns:
            Message for the planner describing whether reset was done and why.
        """
        if not getattr(self, "_uses_plan_checkpoints", False):
            return "Plan checkpoint reset is not available for this agent."
        if self.previous_plan_checkpoint is None or self.previous_checkpoint_scores is None:
            return "No previous checkpoint to reset to."
        if self.checkpoint_scores is None:
            return "No current scores to compare; run a critique first."
        current_total = compute_total_score(self.checkpoint_scores)
        previous_total = compute_total_score(self.previous_checkpoint_scores)
        if current_total >= previous_total:
            return (
                f"Scores did not regress; no reset needed. "
                f"Current total: {current_total}, previous checkpoint: {previous_total}."
            )
        await self._apply_plan_checkpoint_rollback(self.previous_plan_checkpoint)
        console_logger.info(
            "Plan checkpoint reset: reverted to previous version "
            "(current total=%d, previous=%d)",
            current_total,
            previous_total,
        )
        return (
            f"Reset plan to previous checkpoint (scores regressed: "
            f"current total {current_total} vs previous {previous_total}). "
            "The designer session has been updated with the reverted plan."
        )

    async def _get_critique_context_async(self) -> str:
        """Async wrapper for getting critique context.

        Subclasses that need async access to context (e.g. reading from sessions)
        can override this method instead of _get_critique_context().
        """
        return self._get_critique_context()

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Implementation for critique request.

        Runs the critic agent and returns structured critique. For plan-only
        agents, prompt has no physics/placement context. For scene agents,
        optional checkpoint and render-dir handling is applied.

        Args:
            update_checkpoint: Whether to shift checkpoints (only used when
                _uses_scene_checkpoints is True).

        Returns:
            Critique text with optional score deltas for planner.
        """
        console_logger.info("Tool called: request_critique")

        prompt_enum = self._get_critique_prompt_enum()
        extra_kwargs = self._get_extra_critique_kwargs()
        scene = getattr(self, "scene", None)

        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            **extra_kwargs,
        )
        context = await self._get_critique_context_async()
        critic_input = (critique_instruction + "\n\n" + context) if context else critique_instruction
        result = await Runner.run(
            starting_agent=self.critic,
            input=critic_input,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="CRITIC")

        response = result.final_output_as(CritiqueWithScores)
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="CRITIQUE SCORES")

        # Save scores to file when we have a suitable directory.
        scores_dir = None
        rendering_manager = getattr(self, "rendering_manager", None)
        if rendering_manager is not None:
            scores_dir = getattr(rendering_manager, "last_render_dir", None)
        if scores_dir is None:
            scores_dir = getattr(self.logger, "output_dir", None)
        if scores_dir is not None:
            scores_path = Path(scores_dir) / "scores.yaml"
            try:
                with open(scores_path, "w") as f:
                    yaml.dump(
                        scores_to_dict(response),
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                console_logger.info(f"Scores saved to: {scores_path}")
            except OSError as e:
                console_logger.warning(f"Could not save scores to {scores_path}: {e}")

        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        self.previous_scores = response

        if getattr(self, "_uses_scene_checkpoints", False) and scene is not None:
            images_dir = scores_dir if rendering_manager else None
            if update_checkpoint:
                self.previous_scene_checkpoint = self.scene_checkpoint
                self.previous_checkpoint_scores = self.checkpoint_scores
                self.previous_checkpoint_render_dir = self.checkpoint_render_dir
                self.scene_checkpoint = copy.deepcopy(scene.to_state_dict())
                self.checkpoint_scores = response
                self.checkpoint_render_dir = images_dir
                self.checkpoint_scene_hash = scene.content_hash()
            self.final_render_dir = images_dir

        # Plan checkpoints: same N-1/N pattern for plan state (e.g. project plan text).
        if getattr(self, "_uses_plan_checkpoints", False) and update_checkpoint:
            plan_state = self._get_plan_state_for_checkpoint()
            if plan_state is not None:
                self.previous_plan_checkpoint = self.plan_checkpoint
                self.previous_checkpoint_scores = self.checkpoint_scores
                self.plan_checkpoint = copy.deepcopy(plan_state)
                self.checkpoint_scores = response
                plan_text = plan_state.get("plan_text") or ""
                self.checkpoint_plan_hash = hash(plan_text)

        return response.critique + score_change_msg

    @abstractmethod
    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Prompt enum for domain-specific design change instruction.
        """

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Implementation for design change request.

        Args:
            instruction: Specific changes to make based on critique feedback.

        Returns:
            Designer's report of what was changed.
        """
        console_logger.info("Tool called: request_design_change")

        # Get instruction from prompt registry with domain-specific enum.
        prompt_enum = self._get_design_change_prompt_enum()
        full_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            instruction=instruction,
        )

        # Designer run with critique-based instruction.
        result = await Runner.run(
            starting_agent=self.designer,
            input=full_instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

            # Persist the updated design into the designer session so that
            # downstream critics can read the current plan from session
            # history (and benefit from trimming/summarization).
            try:
                if self.designer_session is not None:
                    await self.designer_session.add_items(
                        [
                            {
                                "role": "assistant",
                                "content": result.final_output,
                            }
                        ]
                    )
            except Exception as e:  # Best-effort; logging only.
                console_logger.warning(
                    f"Failed to append design-change output to designer session: {e}"
                )
            self._on_designer_output(result.final_output)

        return result.final_output or ""

    @abstractmethod
    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Prompt enum for domain-specific initial design instruction.
        """

    @abstractmethod
    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dictionary of kwargs to pass to get_prompt() for initial design.
        """

    def _get_context_image_path(self) -> Path | None:
        """Get optional context image path for initial design.

        Subclasses can override to provide an AI-generated reference image
        that will be included in the initial design user message.

        Returns:
            Path to context image, or None if not available.
        """
        return None

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Build the input for initial design request.

        If a context image is available, constructs a multimodal message
        with both text instruction and the reference image.

        Args:
            instruction: Text instruction for the designer.

        Returns:
            Either plain text or a list with a multimodal user message.
        """
        context_image_path = self._get_context_image_path()
        if context_image_path and context_image_path.exists():
            # Build multimodal input with text + image.
            console_logger.info(
                f"Including context image in initial design: {context_image_path}"
            )
            image_base64 = encode_image_to_base64(context_image_path)
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        },
                    ],
                }
            ]
        # No context image - use plain text.
        return instruction

    async def _request_initial_design_impl(self) -> str:
        """Implementation for initial design request.

        Returns:
            Designer's report of initial design.
        """
        console_logger.info("Tool called: request_initial_design")
        try:
            # Get instruction from prompt registry with domain-specific enum and kwargs.
            console_logger.info("Building initial design instruction (get_prompt)...")
            prompt_enum = self._get_initial_design_prompt_enum()
            prompt_kwargs = self._get_initial_design_prompt_kwargs()
            instruction = self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum, **prompt_kwargs
            )
            console_logger.info("Building initial design input (may include image)...")
            # Build input (may include context image if enabled).
            input_message = self._build_initial_design_input(instruction)

            # Designer runs with initial design instruction.
            console_logger.info("Running designer (initial design)...")
            result = await Runner.run(
                starting_agent=self.designer,
                input=input_message,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
                run_config=self._create_run_config(),
            )
            try:
                log_agent_usage(result=result, agent_name="DESIGNER (INITIAL)")
            except Exception as e:
                console_logger.warning("Could not log designer usage (continuing): %s", e)
            has_output = bool(getattr(result, "final_output", None))
            console_logger.info(
                "Designer finished: final_output=%s (len=%s)",
                "yes" if has_output else "no",
                len(result.final_output) if has_output else 0,
            )

            if result.final_output:
                log_agent_response(
                    response=result.final_output, agent_name="DESIGNER (INITIAL)"
                )

                # Persist the initial design into the designer session so that
                # downstream critics can read the plan from session history.
                try:
                    if self.designer_session is not None:
                        await self.designer_session.add_items(
                            [
                                {
                                    "role": "assistant",
                                    "content": result.final_output,
                                }
                            ]
                        )
                except Exception as e:  # Best-effort; logging only.
                    console_logger.warning(
                        f"Failed to append initial design output to designer session: {e}"
                    )
                console_logger.info("Calling _on_designer_output(len=%d) -> extract Mermaid, render PNG/PDF", len(result.final_output))
                self._on_designer_output(result.final_output)
                console_logger.info("_on_designer_output completed")
            else:
                console_logger.warning("Designer returned no final_output; skipping _on_designer_output")

            return result.final_output or ""
        except Exception as e:
            console_logger.exception(
                "request_initial_design failed (designer did not run): %s", e
            )
            return f"(Initial design failed: {e})"
