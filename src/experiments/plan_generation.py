"""Plan-generation experiment: project (and optionally module/task) planning only."""

import asyncio
import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.experiments.base_experiment import BaseExperiment
from src.module_agents.stateful_module_agent import StatefulModuleAgent
from src.project_agents.stateful_project_agent import StatefulProjectAgent
from src.task_agents.stateful_task_agent import StatefulTaskAgent
from src.utils.logging import BaseLogger, FileLoggingContext
from src.visualization_agents.stateful_visualization_agent import (
    StatefulVisualizationAgent,
)
from src.utils.parallel import run_parallel_isolated
from src.utils.print_utils import bold_green, yellow

console_logger = logging.getLogger(__name__)

# Pipeline stages for plan generation (project → module → task → visualization)
PIPELINE_STAGES = ["project", "module", "task", "visualization"]


def _load_prompts_from_csv(csv_path: str) -> list[tuple[int, str]]:
    """Load prompts from CSV.

    Args:
        csv_path: Path to CSV with columns: index, description (header required).

    Returns:
        List of (index, prompt) tuples.
    """
    prompts_with_ids = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"CSV row {row_num} has fewer than 2 columns: {row}")
            try:
                idx = int(row[0])
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: index '{row[0]}' is not a valid integer"
                )
            prompts_with_ids.append((idx, row[1]))
    return prompts_with_ids


def _run_pipeline_stages(
    prompt: str,
    out_path: Path,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
) -> str | None:
    """Run project → module → task → visualization pipeline stages for a single prompt.

    Args:
        prompt: Initial project brief.
        out_path: Output directory for this prompt (e.g. prompt_000).
        cfg_dict: Resolved config as dict.
        start_stage: First stage to run (project, module, task, or visualization).
        stop_stage: Last stage to run (project, module, task, or visualization).

    Returns:
        Final plan text or diagram path from the last stage, or None on failure.
    """
    from omegaconf import OmegaConf

    logger = BaseLogger(output_dir=out_path)
    cfg = OmegaConf.create(cfg_dict)

    project_plan_path = out_path / "project_plan.md"
    module_plan_path = out_path / "module_plan.md"
    task_plan_path = out_path / "task_plan.md"

    stages = PIPELINE_STAGES
    start_idx = stages.index(start_stage)
    stop_idx = stages.index(stop_stage)

    current_input = prompt
    final_plan = ""

    if start_idx <= 0 <= stop_idx:
        # Project stage
        console_logger.info("Running project stage")
        project_agent = BaseExperiment.build_project_agent(
            cfg_dict=cfg_dict,
            compatible_agents=PlanGenerationExperiment.compatible_project_agents,
            logger=logger,
        )
        final_plan = asyncio.run(
            project_agent.generate_project_plan(prompt=current_input, output_dir=out_path)
        )
        current_input = project_plan_path.read_text(encoding="utf-8")

    if start_idx <= 1 <= stop_idx:
        # Module stage: read project_plan.md as input
        if not project_plan_path.exists():
            console_logger.warning(
                "Skipping module stage: project_plan.md not found. Run project stage first."
            )
        else:
            if start_idx == 1:
                current_input = project_plan_path.read_text(encoding="utf-8")
            console_logger.info("Running module stage")
            module_agent = BaseExperiment.build_module_agent(
                cfg_dict=cfg_dict,
                compatible_agents=PlanGenerationExperiment.compatible_module_agents,
                logger=logger,
            )
            final_plan = asyncio.run(
                module_agent.generate_module_plan(
                    prompt=current_input, output_dir=out_path
                )
            )
            current_input = module_plan_path.read_text(encoding="utf-8")

    if start_idx <= 2 <= stop_idx:
        # Task stage: read module_plan.md as input
        if not module_plan_path.exists():
            console_logger.warning(
                "Skipping task stage: module_plan.md not found. Run module stage first."
            )
        else:
            if start_idx == 2:
                current_input = module_plan_path.read_text(encoding="utf-8")
            console_logger.info("Running task stage")
            task_agent = BaseExperiment.build_task_agent(
                cfg_dict=cfg_dict,
                compatible_agents=PlanGenerationExperiment.compatible_task_agents,
                logger=logger,
            )
            final_plan = asyncio.run(
                task_agent.generate_task_plan(
                    prompt=current_input, output_dir=out_path
                )
            )
            if task_plan_path.exists():
                console_logger.info(f"Task plan written to: {task_plan_path}")

    if start_idx <= 3 <= stop_idx:
        # Visualization stage: read all three plan files, produce hierarchy diagram
        if not all(
            p.exists()
            for p in (project_plan_path, module_plan_path, task_plan_path)
        ):
            console_logger.warning(
                "Skipping visualization stage: project_plan.md, module_plan.md, or "
                "task_plan.md not found. Run project, module, and task stages first."
            )
        else:
            console_logger.info("Running visualization stage")
            console_logger.info("[VIS] Step 0: Building visualization agent; output_dir=%s", out_path)
            visualization_agent = BaseExperiment.build_visualization_agent(
                cfg_dict=cfg_dict,
                compatible_agents=PlanGenerationExperiment.compatible_visualization_agents,
                logger=logger,
            )
            console_logger.info("[VIS] Step 0b: Calling generate_hierarchy_diagram(output_dir=%s)", out_path)
            diagram_result = asyncio.run(
                visualization_agent.generate_hierarchy_diagram(output_dir=out_path)
            )
            console_logger.info("[VIS] Step 0c: generate_hierarchy_diagram returned: %r", diagram_result)
            if diagram_result:
                final_plan = str(diagram_result)
                console_logger.info(f"Hierarchy diagram written to: {diagram_result}")

    return final_plan


def _generate_project_plan_worker(
    prompt: str,
    index: int,
    output_dir: str,
    cfg_dict: dict,
    experiment_run_id: str | None = None,
) -> str | None:
    """Generate plans for a single prompt (project → module → task pipeline).

    Top-level function for pickling when using ProcessPoolExecutor.
    All arguments must be picklable.

    Args:
        prompt: Project description.
        index: Prompt index for directory naming.
        output_dir: Base output directory (string path).
        cfg_dict: Resolved config as dict.
        experiment_run_id: Optional run ID for logging.

    Returns:
        Final plan text on success, None on failure (exception is raised).
    """
    out_path = Path(output_dir) / f"prompt_{index:03d}"
    out_path.mkdir(parents=True, exist_ok=True)

    log_path = out_path / "plan.log"
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(f"Pipeline worker started for prompt {index:03d}")

        pipeline_cfg = cfg_dict.get("experiment", {}).get("pipeline", {})
        start_stage = pipeline_cfg.get("start_stage", "project")
        stop_stage = pipeline_cfg.get("stop_stage", "project")

        plan = _run_pipeline_stages(
            prompt=prompt,
            out_path=out_path,
            cfg_dict=cfg_dict,
            start_stage=start_stage,
            stop_stage=stop_stage,
        )
        console_logger.info(f"Pipeline worker completed for prompt {index:03d}")
        return plan


class PlanGenerationExperiment(BaseExperiment):
    """Experiment that runs project, module, task, and optionally visualization stages."""

    compatible_project_agents = {
        "workflow_project_agent": StatefulProjectAgent,
    }
    compatible_module_agents = {
        "workflow_module_agent": StatefulModuleAgent,
    }
    compatible_task_agents = {
        "workflow_task_agent": StatefulTaskAgent,
    }
    compatible_visualization_agents = {
        "workflow_visualization_agent": StatefulVisualizationAgent,
    }

    def _run_serial(self, prompts_with_ids: list[tuple[int, str]], cfg_dict: dict) -> None:
        """Run project → module → task pipeline sequentially."""
        pipeline_cfg = cfg_dict.get("experiment", {}).get("pipeline", {})
        start_stage = pipeline_cfg.get("start_stage", "project")
        stop_stage = pipeline_cfg.get("stop_stage", "project")
        console_logger.info(
            f"Running pipeline serially (stages: {start_stage} → {stop_stage})"
        )

        for index, prompt in prompts_with_ids:
            out_dir = self.output_dir / f"prompt_{index:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            _run_pipeline_stages(
                prompt=prompt,
                out_path=out_dir,
                cfg_dict=cfg_dict,
                start_stage=start_stage,
                stop_stage=stop_stage,
            )
            console_logger.info(f"Completed prompt {index:03d}")

    def _run_parallel(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        num_workers: int,
        experiment_run_id: str,
    ) -> None:
        """Run project planning in parallel processes."""
        console_logger.info(f"Running project planning in parallel with {num_workers} workers")

        tasks = []
        for index, prompt in prompts_with_ids:
            task_id = f"prompt_{index:03d}"
            tasks.append(
                (
                    task_id,
                    _generate_project_plan_worker,
                    {
                        "prompt": prompt,
                        "index": index,
                        "output_dir": str(self.output_dir),
                        "cfg_dict": cfg_dict,
                        "experiment_run_id": experiment_run_id,
                    },
                )
            )
            console_logger.info(f"Queued {task_id}: {prompt[:60]}...")

        results = run_parallel_isolated(tasks=tasks, max_workers=num_workers)

        failed = [(tid, err) for tid, (ok, err) in results.items() if not ok]
        if failed:
            details = "\n".join(f"  - {tid}: {err}" for tid, err in failed)
            raise RuntimeError(f"{len(failed)}/{len(tasks)} prompt(s) failed:\n{details}")

    def plan_project(self) -> None:
        """Run project planning for all configured prompts (serial or parallel)."""
        pipeline_cfg = self.cfg.experiment.pipeline
        start_stage = pipeline_cfg.start_stage
        stop_stage = pipeline_cfg.stop_stage

        if start_stage not in PIPELINE_STAGES or stop_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid pipeline stages. start_stage={start_stage!r}, stop_stage={stop_stage!r}. "
                f"Valid: {PIPELINE_STAGES}"
            )
        if PIPELINE_STAGES.index(start_stage) > PIPELINE_STAGES.index(stop_stage):
            raise ValueError(
                f"start_stage '{start_stage}' cannot be after stop_stage '{stop_stage}'"
            )

        console_logger.info(
            f"Pipeline stages: {start_stage} → {stop_stage}"
        )

        csv_path = self.cfg.experiment.get("csv_path")
        if csv_path:
            prompts_with_ids = _load_prompts_from_csv(csv_path)
            console_logger.info(f"Loaded {len(prompts_with_ids)} prompts from CSV: {csv_path}")
        else:
            prompts = self.cfg.experiment.prompts
            prompts_with_ids = list(enumerate(prompts))

        num_workers = min(self.cfg.experiment.num_workers, len(prompts_with_ids))
        experiment_run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        console_logger.info(f"Starting project planning: {num_workers} worker(s), {len(prompts_with_ids)} prompt(s)")
        console_logger.info(f"Experiment run ID: {experiment_run_id}")

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        if num_workers == 1:
            self._run_serial(prompts_with_ids=prompts_with_ids, cfg_dict=cfg_dict)
        else:
            self._run_parallel(
                prompts_with_ids=prompts_with_ids,
                cfg_dict=cfg_dict,
                num_workers=num_workers,
                experiment_run_id=experiment_run_id,
            )

        console_logger.info("All pipeline runs completed")
        console_logger.info("=" * 60)
        console_logger.info(bold_green("ALL PIPELINE RUNS COMPLETED!"))
        console_logger.info("=" * 60)
        console_logger.info(yellow("Outputs saved under: ") + str(self.output_dir))
        console_logger.info("=" * 60)
