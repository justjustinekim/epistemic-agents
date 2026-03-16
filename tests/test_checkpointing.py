"""Tests for debate checkpointing (Change 11)."""
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone

from epistemic_agents.schema import DebateCheckpoint, PanelSynthesis, ProviderPosition
from epistemic_agents.orchestrator import Orchestrator


def test_checkpoint_model_creation():
    cp = DebateCheckpoint(task="test task", phase=1, phase_name="debate")
    assert cp.task == "test task"
    assert cp.phase == 1
    assert cp.synthesis is None
    assert cp.rounds == []


def test_checkpoint_serialization_roundtrip():
    cp = DebateCheckpoint(
        task="test task", phase=2, phase_name="synthesis",
        rounds=[{"test": "data"}],
    )
    data = cp.model_dump(mode="json")
    restored = DebateCheckpoint.model_validate(data)
    assert restored.task == "test task"
    assert restored.phase == 2


def test_checkpoint_path_deterministic():
    path1 = Orchestrator._checkpoint_path("test task", 1)
    path2 = Orchestrator._checkpoint_path("test task", 1)
    assert path1 == path2


def test_checkpoint_path_different_phases():
    path1 = Orchestrator._checkpoint_path("test task", 1)
    path2 = Orchestrator._checkpoint_path("test task", 2)
    assert path1 != path2


def test_save_and_load_checkpoint():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            orch = Orchestrator()
            cp = DebateCheckpoint(task="test task", phase=2, phase_name="synthesis")
            orch._save_checkpoint(cp)
            loaded = Orchestrator.load_checkpoint("test task")
            assert loaded is not None
            assert loaded.phase == 2
            assert loaded.phase_name == "synthesis"
        finally:
            os.chdir(old_cwd)


def test_load_checkpoint_none_when_missing():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            loaded = Orchestrator.load_checkpoint("nonexistent task")
            assert loaded is None
        finally:
            os.chdir(old_cwd)


def test_load_returns_latest_phase():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            orch = Orchestrator()
            orch._save_checkpoint(DebateCheckpoint(task="t", phase=1, phase_name="debate"))
            orch._save_checkpoint(DebateCheckpoint(task="t", phase=3, phase_name="refutation"))
            loaded = Orchestrator.load_checkpoint("t")
            assert loaded is not None
            assert loaded.phase == 3
        finally:
            os.chdir(old_cwd)


def test_checkpoint_with_synthesis():
    cp = DebateCheckpoint(
        task="test",
        phase=4,
        phase_name="resynthesis",
        synthesis=PanelSynthesis(
            task="test",
            provider_positions=[],
            synthesized_strategy="do X",
            meta_confidence="moderate",
        ),
    )
    data = cp.model_dump(mode="json")
    restored = DebateCheckpoint.model_validate(data)
    assert restored.synthesis is not None
    assert restored.synthesis.synthesized_strategy == "do X"
