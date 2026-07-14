"""Tests for rescoring trajectories with tau2.scripts.evaluate_trajectories.

Regression coverage for full-duplex (voice) results: rescoring must detect
the communication mode from the results and use the tick-based evaluators,
instead of silently evaluating simulation.messages with the half-duplex
evaluators (reported in PR #386).
"""

from tau2.data_model.message import Tick, ToolCall, ToolMessage
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.orchestrator.modes import CommunicationMode
from tau2.run import get_tasks
from tau2.scripts import evaluate_trajectories as evaluate_trajectories_module
from tau2.scripts.evaluate_trajectories import (
    compute_simulation_rewards,
    get_communication_mode,
)

# ---- Fixtures ----


def _make_info(
    user_implementation: str = "user_simulator",
    audio_native_config: AudioNativeConfig = None,
) -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation=user_implementation),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
        audio_native_config=audio_native_config,
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_half_duplex_sim(task_id: str) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )


def _make_full_duplex_sim(task_id: str, ticks: list[Tick] = None) -> SimulationRun:
    """Full-duplex sims store ticks and no messages (see FullDuplexOrchestrator)."""
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=None,
        ticks=ticks if ticks is not None else [],
        mode=CommunicationMode.FULL_DUPLEX.value,
    )


# ---- Mode detection ----


class TestGetCommunicationMode:
    def test_half_duplex_by_default(self):
        sim = _make_half_duplex_sim("t0")
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.HALF_DUPLEX

    def test_detects_voice_streaming_user_implementation(self):
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[_make_task("t0")],
            simulations=[],
        )
        assert get_communication_mode(results) == CommunicationMode.FULL_DUPLEX

    def test_detects_audio_native_config(self):
        results = Results(
            info=_make_info(audio_native_config=AudioNativeConfig()),
            tasks=[_make_task("t0")],
            simulations=[],
        )
        assert get_communication_mode(results) == CommunicationMode.FULL_DUPLEX

    def test_detects_full_duplex_simulation_mode(self):
        # Isolate the mode-field signal: no ticks stored (e.g. saved without
        # verbose tick data) and no run-level voice info.
        sim = _make_full_duplex_sim("t0")
        sim.ticks = None
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.FULL_DUPLEX

    def test_detects_ticks_when_mode_missing(self):
        # Older full-duplex trajectories may predate the SimulationRun.mode
        # field and deserialize with the half_duplex default.
        sim = _make_full_duplex_sim("t0")
        sim.mode = CommunicationMode.HALF_DUPLEX.value
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.FULL_DUPLEX

    def test_mixed_results_detected_per_simulation(self):
        full_duplex_sim = _make_full_duplex_sim("t0")
        half_duplex_sim = _make_half_duplex_sim("t1")
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0"), _make_task("t1")],
            simulations=[full_duplex_sim, half_duplex_sim],
        )
        assert (
            get_communication_mode(results, full_duplex_sim)
            == CommunicationMode.FULL_DUPLEX
        )
        assert (
            get_communication_mode(results, half_duplex_sim)
            == CommunicationMode.HALF_DUPLEX
        )


# ---- Rescoring passes the detected mode to the evaluator ----


class TestComputeSimulationRewardsMode:
    def _capture_mode(self, monkeypatch, results) -> list:
        captured = []

        def fake_evaluate_simulation(**kwargs):
            captured.append(kwargs)
            return RewardInfo(reward=1.0)

        monkeypatch.setattr(
            evaluate_trajectories_module,
            "evaluate_simulation",
            fake_evaluate_simulation,
        )
        compute_simulation_rewards(results)
        return captured

    def test_full_duplex_results_evaluated_in_full_duplex_mode(self, monkeypatch):
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[_make_task("t0")],
            simulations=[_make_full_duplex_sim("t0")],
        )
        captured = self._capture_mode(monkeypatch, results)
        assert len(captured) == 1
        assert captured[0]["mode"] == CommunicationMode.FULL_DUPLEX

    def test_half_duplex_results_evaluated_in_half_duplex_mode(self, monkeypatch):
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_mode(monkeypatch, results)
        assert len(captured) == 1
        assert captured[0]["mode"] == CommunicationMode.HALF_DUPLEX


# ---- End-to-end: rescoring a full-duplex results file uses tick evaluators ----


class TestFullDuplexRescoring:
    def test_rescoring_full_duplex_results_uses_ticks(self):
        """Regression test for PR #386: rescoring full-duplex results must
        evaluate simulation.ticks. The golden create_task action below only
        exists in the ticks (messages is None, as in real voice trajectories),
        so the half-duplex evaluators cannot produce this reward."""
        task = get_tasks("mock", task_ids=["create_task_1"])[0]
        tick = Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:30",
            agent_tool_calls=[
                ToolCall(
                    id="call_1",
                    name="create_task",
                    arguments={"user_id": "user_1", "title": "Important Meeting"},
                )
            ],
            agent_tool_results=[
                ToolMessage(
                    id="call_1",
                    role="tool",
                    content='{"task_id": "task_2", "title": "Important Meeting", '
                    '"description": null, "status": "pending"}',
                    requestor="assistant",
                )
            ],
        )
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[task],
            simulations=[_make_full_duplex_sim(task.id, ticks=[tick])],
        )

        rescored = compute_simulation_rewards(results)

        reward_info = rescored.simulations[0].reward_info
        assert reward_info.reward == 1.0
        assert reward_info.db_check is not None
        assert reward_info.db_check.db_match is True
        # The tick-based action evaluator matched the create_task call.
        assert reward_info.action_checks is not None
        assert all(check.action_match for check in reward_info.action_checks)
