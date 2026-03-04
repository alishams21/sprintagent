"""Structured scoring and critique utilities for design evaluation.

This module provides data structures and utilities for critic-based evaluation
with categorical scoring. Includes score computation, delta tracking, and formatted
output for iterative design improvement.
"""

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CategoryScore:
    """Score for a single evaluation category.

    Used by both furniture and manipuland agents for structured critique output.
    """

    name: str
    """Category name (e.g., 'realism', 'functionality') - for logging."""
    grade: int
    """Score from 0-10."""
    comment: str
    """Brief justification for the score."""


@dataclass
class CritiqueWithScores(ABC):
    """Base class for agent-specific critique scoring.

    Subclasses define agent-specific categories and implement get_scores()
    for generic utility function access.
    """

    critique: str
    """Natural language critique for the planner."""

    @abstractmethod
    def get_scores(self) -> list[CategoryScore]:
        """Return all category scores for generic processing.

        Returns:
            List of CategoryScore objects for all categories.
        """


@dataclass
class ProjectCritiqueWithScores(CritiqueWithScores):
    """Project plan critic with five evaluation dimensions (0–10 each).

    Matches the categories in project_agent critic prompts.
    """

    clarity_structure: CategoryScore
    """Clarity and structure of project → modules → tasks hierarchy."""
    feasibility_scope: CategoryScore
    """Feasibility and scope (realism, task sizing)."""
    risk_dependencies: CategoryScore
    """Risk and dependency management."""
    execution_readiness: CategoryScore
    """Execution readiness (testing, observability, rollout)."""
    prompt_following: CategoryScore
    """Adherence to project brief requirements."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all project critique category scores."""
        return [
            self.clarity_structure,
            self.feasibility_scope,
            self.risk_dependencies,
            self.execution_readiness,
            self.prompt_following,
        ]


@dataclass
class ModuleCritiqueWithScores(CritiqueWithScores):
    """Module plan critic with five evaluation dimensions (0–10 each).

    Matches the categories in module_agent critic prompts.
    """

    clarity_structure: CategoryScore
    """Clarity and structure of module purpose, boundaries, organization."""
    technical_soundness: CategoryScore
    """Technical soundness and feasibility of structure/approach."""
    risk_dependencies: CategoryScore
    """Risk and dependency management."""
    execution_readiness: CategoryScore
    """Execution readiness (actionable tasks, definitions of done)."""
    prompt_following: CategoryScore
    """Adherence to module brief requirements."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all module critique category scores."""
        return [
            self.clarity_structure,
            self.technical_soundness,
            self.risk_dependencies,
            self.execution_readiness,
            self.prompt_following,
        ]


@dataclass
class TaskCritiqueWithScores(CritiqueWithScores):
    """Task plan critic with five evaluation dimensions (0–10 each).

    Matches the categories in task_agent critic prompts.
    """

    clarity: CategoryScore
    """Clarity and understandability for assignee."""
    scope_feasibility: CategoryScore
    """Scope and feasibility (right-sized, realistic)."""
    coverage: CategoryScore
    """Coverage of subtasks/steps and criteria."""
    execution_readiness: CategoryScore
    """Execution readiness (can start immediately)."""
    prompt_following: CategoryScore
    """Adherence to task brief and context."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all task critique category scores."""
        return [
            self.clarity,
            self.scope_feasibility,
            self.coverage,
            self.execution_readiness,
            self.prompt_following,
        ]


@dataclass
class VisualizationCritiqueWithScores(CritiqueWithScores):
    """Visualization (hierarchy diagram) critic with consistency and completeness.

    Used when the critic evaluates a diagram image against project/module/task plans.
    """

    hierarchy_consistency: CategoryScore
    """Project components on top, modules under components, tasks under modules."""
    completeness: CategoryScore
    """All components, modules, and tasks from the plans appear in the diagram."""
    clarity: CategoryScore
    """Diagram is readable and labels match plan names."""
    layout_quality: CategoryScore
    """Layout is logical and dependencies are clear."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all visualization critique category scores."""
        return [
            self.hierarchy_consistency,
            self.completeness,
            self.clarity,
            self.layout_quality,
        ]


def compute_total_score(scores: CritiqueWithScores) -> int:
    """Compute total score across all categories.

    Args:
        scores: Critique scores containing all categories.

    Returns:
        Sum of all category grades (range depends on number of categories).
    """
    return sum(score.grade for score in scores.get_scores())


def compute_score_deltas(
    current: CritiqueWithScores, previous: CritiqueWithScores
) -> dict[str, int]:
    """Compute per-category score changes.

    Args:
        current: Current critique scores.
        previous: Previous critique scores.

    Returns:
        Dictionary mapping category names to score changes (can be negative).
    """
    current_scores = {s.name: s.grade for s in current.get_scores()}
    previous_scores = {s.name: s.grade for s in previous.get_scores()}

    return {
        name: current_scores[name] - previous_scores[name] for name in current_scores
    }


def scores_to_dict(scores: CritiqueWithScores) -> dict[str, dict[str, Any]]:
    """Convert scores to YAML-serializable dictionary.

    Args:
        scores: Critique scores to convert.

    Returns:
        Dictionary with category scores and summary, suitable for YAML output.
    """
    result = {
        score.name: {
            "grade": score.grade,
            "comment": score.comment,
        }
        for score in scores.get_scores()
    }
    result["summary"] = scores.critique
    return result


def log_critique_scores(
    scores: CritiqueWithScores, title: str = "CRITIQUE SCORES"
) -> None:
    """Log critique scores with consistent formatting.

    Args:
        scores: Critique scores to log.
        title: Title to display in log header.
    """
    console_logger = logging.getLogger(__name__)
    console_logger.info("=" * 60)
    console_logger.info(title)
    console_logger.info("=" * 60)
    for score in scores.get_scores():
        console_logger.info(f"{score.name.replace('_', ' ').title()}: {score.grade}/10")
        console_logger.info(f"  {score.comment}")
    console_logger.info("=" * 60)


def format_score_deltas_for_planner(
    current_scores: CritiqueWithScores,
    previous_scores: CritiqueWithScores,
    format_style: str = "detailed",
) -> str:
    """Format score changes for planner agent feedback.

    Args:
        current_scores: Current critique scores.
        previous_scores: Previous critique scores.
        format_style: Formatting style - "detailed" (with transitions) or
            "compact" (deltas only).

    Returns:
        Formatted string showing score changes.
    """
    deltas = compute_score_deltas(current_scores, previous_scores)
    sum_delta = sum(deltas.values())

    current_by_name = {s.name: s for s in current_scores.get_scores()}
    previous_by_name = {s.name: s for s in previous_scores.get_scores()}

    if format_style == "detailed":
        max_drop = abs(min(min(deltas.values(), default=0), 0))
        lines = ["\n\n**Score Changes:**"]
        for name in deltas:
            display_name = name.replace("_", " ").title()
            prev_grade = previous_by_name[name].grade
            curr_grade = current_by_name[name].grade
            lines.append(
                f"{display_name}: {prev_grade}→{curr_grade} ({deltas[name]:+d})"
            )
        lines.append(f"**Total: {sum_delta:+d}** (Max drop: {max_drop})")
        return "\n".join(lines)
    else:  # compact
        delta_lines = [
            f"- {name.replace('_', ' ').title()}: {delta:+d}"
            for name, delta in deltas.items()
        ]
        return (
            f"\n\n## Score Changes from Previous Iteration\n"
            f"{chr(10).join(delta_lines)}\n"
            f"- **Total: {sum_delta:+d}**"
        )


def log_agent_response(response: str, agent_name: str) -> None:
    """Log agent response with consistent formatting.

    Args:
        response: The agent's final response text.
        agent_name: Name of the agent (e.g., "PLANNER", "DESIGNER").
    """
    console_logger = logging.getLogger(__name__)
    console_logger.info("=" * 60)
    console_logger.info(f"{agent_name} RESPONSE")
    console_logger.info("=" * 60)
    console_logger.info(response)
    console_logger.info("=" * 60)
