"""Stateful visualization agent using planner/designer/critic workflow with VLM critic."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from agents import Agent, FunctionTool, Runner, RunResult
from omegaconf import DictConfig

from src.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    AgentType,
    log_agent_usage,
)
from src.agent_utils.scoring import (
    VisualizationCritiqueWithScores,
    compute_total_score,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from src.prompts import VisualizationAgentPrompts, prompt_registry
from src.utils.openai import encode_image_to_base64
from src.visualization_agents.base_visualization_agent import BaseVisualizationAgent
from src.utils.mermaid_render import (
    extract_mermaid_from_text,
    render_mermaid_to_png,
    sanitize_mermaid_for_render,
    render_mermaid_to_pdf,
)

console_logger = logging.getLogger(__name__)


class StatefulVisualizationAgent(BaseStatefulAgent, BaseVisualizationAgent):
    """Stateful visualization agent: produces hierarchy diagram from plan MD files.

    Designer outputs Mermaid; we render to PNG/PDF. Critic uses VLM to compare
    diagram image against project/module/task plans.
    """

    _uses_plan_checkpoints: bool = True

    def __init__(self, cfg: DictConfig, logger: Any):
        BaseVisualizationAgent.__init__(self, cfg=cfg, logger=logger)
        BaseStatefulAgent.__init__(self, cfg=cfg, logger=logger)

        output_dir_name = (
            Path(self.logger.output_dir).name
            if getattr(self.logger, "output_dir", None) is not None
            else ""
        )
        base = f"{output_dir_name}_" if output_dir_name else ""
        self.session_prefix: str = f"{base}visualization_"

        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=self.session_prefix
        )

        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None

        self.project_plan_text: str = ""
        self.module_plan_text: str = ""
        self.task_plan_text: str = ""
        self.current_plan_name: str = ""  # current plan in per-plan loop (project/module/task)
        self.current_plan_content: str = ""
        self.current_mermaid_text: str = ""
        self.current_diagram_path: Path | None = None
        self.current_plan_text: str = ""  # designer raw output for checkpoint
        self._diagram_output_dir: Path | None = None  # set in generate_hierarchy_diagram

    @property
    def agent_type(self) -> AgentType:
        return AgentType.VISUALIZATION

    def _get_final_scores_directory(self) -> Path:
        return Path(self.logger.output_dir) / "final_visualization"

    def _get_critique_prompt_enum(self) -> Any:
        return VisualizationAgentPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_design_change_prompt_enum(self) -> Any:
        return VisualizationAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        return {
            "plan_name": self.current_plan_name,
            "plan_content": self.current_plan_content,
        }

    async def _get_critique_context_async(self) -> str:
        """Context is provided via _get_extra_critique_kwargs and image in overridden critic call."""
        return ""

    def _get_initial_design_prompt_enum(self) -> Any:
        return VisualizationAgentPrompts.DESIGNER_AGENT

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {
            "plan_name": self.current_plan_name,
            "plan_content": self.current_plan_content,
        }

    def _on_designer_output(self, output: str) -> None:
        """Extract Mermaid, render to PNG/PDF, and store paths for critic and checkpoint."""
        console_logger.info("[VIS] _on_designer_output called; output len=%d", len(output or ""))
        self.current_plan_text = output or ""
        output_dir = self._diagram_output_dir if self._diagram_output_dir is not None else Path(self.logger.output_dir)
        base_name = f"{self.current_plan_name}_diagram" if self.current_plan_name else "hierarchy_diagram"
        raw_path = output_dir / f"{self.current_plan_name}_designer_last_output.txt" if self.current_plan_name else output_dir / "visualization_designer_last_output.txt"
        raw_path.write_text(self.current_plan_text, encoding="utf-8")
        console_logger.info("[VIS] Designer raw output saved to %s (%d bytes)", raw_path, len(self.current_plan_text))

        mermaid = extract_mermaid_from_text(output or "")
        if not mermaid:
            console_logger.warning(
                "[VIS] No Mermaid diagram found in designer output; skipping render. Inspect %s",
                raw_path,
            )
            return
        console_logger.info("[VIS] Mermaid extracted (len=%d); rendering PNG and PDF to %s", len(mermaid), output_dir)
        self.current_mermaid_text = mermaid
        mermaid_sanitized = sanitize_mermaid_for_render(mermaid)
        mmd_path = output_dir / f"{base_name}.mmd"
        mmd_path.write_text(mermaid_sanitized, encoding="utf-8")
        png_path = render_mermaid_to_png(mermaid_sanitized, output_dir, f"{base_name}.png")
        self.current_diagram_path = png_path
        if getattr(self.cfg, "render_pdf", True):
            render_mermaid_to_pdf(mermaid_sanitized, output_dir, f"{base_name}.pdf")
        console_logger.info("[VIS] Diagram rendered to %s; png_path=%s", output_dir, png_path)

    def _get_plan_state_for_checkpoint(self) -> dict[str, Any] | None:
        if not self.current_mermaid_text and not self.current_plan_text:
            return None
        return {
            "plan_text": self.current_plan_text,
            "mermaid_text": self.current_mermaid_text,
            "diagram_path": str(self.current_diagram_path) if self.current_diagram_path else None,
        }

    async def _apply_plan_checkpoint_rollback(
        self, plan_checkpoint: dict[str, Any]
    ) -> None:
        plan_text = plan_checkpoint.get("plan_text") or ""
        mermaid_text = plan_checkpoint.get("mermaid_text") or ""
        self.current_plan_text = plan_text
        self.current_mermaid_text = mermaid_text
        output_dir = self._diagram_output_dir if self._diagram_output_dir is not None else Path(self.logger.output_dir)
        base_name = f"{self.current_plan_name}_diagram" if self.current_plan_name else "hierarchy_diagram"
        if mermaid_text:
            png_path = render_mermaid_to_png(mermaid_text, output_dir, f"{base_name}.png")
            self.current_diagram_path = png_path
        rollback_msg = (
            "Resetting to previous diagram version (scores regressed). "
            "The diagram below is the reverted checkpoint."
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
                "Failed to append diagram rollback to designer session: %s", e
            )

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Run critic with diagram image (VLM) + plan texts. Overrides base to pass multimodal input."""
        console_logger.info("[VIS] Tool called: request_critique")

        if self.current_diagram_path is None or not Path(self.current_diagram_path).exists():
            console_logger.warning("[VIS] request_critique: No diagram available (current_diagram_path=%s)", self.current_diagram_path)
            return (
                "No diagram available to critique yet. "
                "Call request_initial_design first to generate the initial diagram."
            )
        console_logger.info("[VIS] Running critic with diagram %s", self.current_diagram_path)

        prompt_enum = self._get_critique_prompt_enum()
        extra = self._get_extra_critique_kwargs()
        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum, **extra
        )
        image_base64 = encode_image_to_base64(self.current_diagram_path)
        critic_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": critique_instruction},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                ],
            }
        ]

        result = await Runner.run(
            starting_agent=self.critic,
            input=critic_input,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="CRITIC")

        response = result.final_output_as(VisualizationCritiqueWithScores)
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="CRITIQUE SCORES (VISUALIZATION)")

        scores_dir = getattr(self.logger, "output_dir", None)
        if scores_dir is not None:
            plan_suffix = f"_{self.current_plan_name}" if self.current_plan_name else ""
            scores_path = Path(scores_dir) / f"visualization_scores{plan_suffix}.yaml"
            try:
                with open(scores_path, "w") as f:
                    yaml.dump(
                        scores_to_dict(response),
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                console_logger.info(f"Visualization scores saved to: {scores_path}")
            except OSError as e:
                console_logger.warning(f"Could not save visualization scores: {e}")

        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )
        self.previous_scores = response

        if getattr(self, "_uses_plan_checkpoints", False) and update_checkpoint:
            plan_state = self._get_plan_state_for_checkpoint()
            if plan_state is not None:
                self.previous_plan_checkpoint = self.plan_checkpoint
                self.previous_checkpoint_scores = self.checkpoint_scores
                self.plan_checkpoint = copy.deepcopy(plan_state)
                self.checkpoint_scores = response
                self.checkpoint_plan_hash = hash(plan_state.get("mermaid_text", ""))

        return response.critique + score_change_msg

    def _create_designer_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        if tools is None:
            tools = []
        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=VisualizationAgentPrompts.DESIGNER_AGENT,
            plan_name=self.current_plan_name,
            plan_content=self.current_plan_content,
        )

    def _create_critic_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        if tools is None:
            tools = []
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=VisualizationAgentPrompts.CRITIC_AGENT,
            output_type=VisualizationCritiqueWithScores,
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        return super()._create_planner_tools()

    def _create_planner_agent(self, tools: list[FunctionTool]) -> Agent:
        max_rounds = getattr(self.cfg, "max_critique_rounds", 2)
        early_finish_min_score = getattr(self.cfg, "early_finish_min_score", 8)
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=VisualizationAgentPrompts.PLANNER_AGENT,
            max_critique_rounds=max_rounds,
            early_finish_min_score=early_finish_min_score,
        )

    async def generate_hierarchy_diagram(self, output_dir: Path) -> Path | str:
        """Generate one Mermaid diagram (and PNG/PDF) per plan file: project, module, task.

        For each plan, runs the full planner/designer/critic loop (same as other agents):
        - Planner is given runner instruction with plan_name; it calls request_initial_design(),
          then request_critique(), then optionally request_design_change() until satisfied.
        - Designer produces Mermaid from the plan; critic (VLM) evaluates diagram image vs plan.
        - Output: project_diagram.*, module_diagram.*, task_diagram.* in output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._diagram_output_dir = output_dir

        project_path = output_dir / "project_plan.md"
        module_path = output_dir / "module_plan.md"
        task_path = output_dir / "task_plan.md"
        if not project_path.exists():
            raise FileNotFoundError(f"Missing {project_path}; run project stage first.")
        if not module_path.exists():
            raise FileNotFoundError(f"Missing {module_path}; run module stage first.")
        if not task_path.exists():
            raise FileNotFoundError(f"Missing {task_path}; run task stage first.")

        plans = [
            ("project", project_path.read_text(encoding="utf-8")),
            ("module", module_path.read_text(encoding="utf-8")),
            ("task", task_path.read_text(encoding="utf-8")),
        ]
        console_logger.info(
            "[VIS] Generating one diagram per plan (planner/designer/critic loop): project, module, task",
        )

        generated: list[str] = []
        planner_max_turns = getattr(
            getattr(self.cfg.agents, "planner_agent", None), "max_turns", 64
        )

        for plan_name, plan_content in plans:
            console_logger.info("[VIS] Starting planner/designer/critic workflow for plan: %s", plan_name)
            self.current_plan_name = plan_name
            self.current_plan_content = plan_content
            # Reset checkpoint state so each plan starts fresh
            self.previous_scores = None
            self.previous_plan_checkpoint = None
            self.plan_checkpoint = None
            self.checkpoint_scores = None
            self.previous_checkpoint_scores = None
            self.checkpoint_plan_hash = None
            self.current_mermaid_text = ""
            self.current_plan_text = ""
            self.current_diagram_path = None

            # New sessions per plan so designer/critic conversation is isolated
            self.designer_session, self.critic_session = self._create_sessions(
                session_prefix=f"{self.session_prefix}{plan_name}_"
            )

            self.designer = self._create_designer_agent()
            self.critic = self._create_critic_agent()
            planner_tools = self._create_planner_tools()
            self.planner = self._create_planner_agent(tools=planner_tools)

            runner_instruction = prompt_registry.get_prompt(
                prompt_enum=VisualizationAgentPrompts.PLANNER_RUNNER_INSTRUCTION,
                plan_name=plan_name,
            )

            result: RunResult = await Runner.run(
                starting_agent=self.planner,
                input=runner_instruction,
                max_turns=planner_max_turns,
            )

            log_agent_usage(result=result, agent_name=f"PLANNER (VIS {plan_name})")

            if self.current_diagram_path and Path(self.current_diagram_path).exists():
                generated.append(str(self.current_diagram_path))
                console_logger.info("[VIS] Completed %s diagram: %s", plan_name, self.current_diagram_path)
            else:
                console_logger.warning("[VIS] No diagram produced for plan: %s", plan_name)

        if generated:
            console_logger.info("[VIS] Diagrams written to %s", output_dir)
            return generated[0] if len(generated) == 1 else str(output_dir)
        return str(output_dir)
