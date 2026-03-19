from epistemic_agents.schema import (
    Belief,
    BeliefGrounding,
    ChallengedBelief,
    ConfidenceLevel,
    CONFIDENCE_LEVEL_TO_SCORE,
    DecisionBoundary,
    Escalation,
    EscalationSeverity,
    StrategicHandoff,
    EscalationType,
    ExecutorFeedback,
    AmendmentType,
    ThinkerAmendment,
    ConversationEntry,
    ConversationLog,
    ProviderPosition,
    AgreementPoint,
    TensionPoint,
    BlindSpot,
    UniqueInsight,
    PanelSynthesis,
    score_to_confidence_level,
    VerificationMethod,
    Verdict,
    PanelResponse,
    DebatePlan,
    DebateCheckpoint,
)
from epistemic_agents.thinker import Thinker
from epistemic_agents.executor import Executor
from epistemic_agents.loop import EpistemicLoop
from epistemic_agents.providers import (
    BaseProvider,
    ClaudeProvider,
    OpenAICompatProvider,
    GeminiProvider,
    VirtualPanelist,
    create_virtual_panelists,
    CodeExecutorProvider,
)
from epistemic_agents.ledger import BeliefLedger, BeliefOutcome, BeliefRecord
from epistemic_agents.bis import importance_scores, rank_beliefs, cascade_falsify, detect_cycles, topological_sort
from epistemic_agents.client import CallUsage, CallCostTracker, get_last_usage
from epistemic_agents.config import get_available_providers, provider_credit_status
from epistemic_agents.panel import ModelPanel
from epistemic_agents.synthesizer import Synthesizer
from epistemic_agents.orchestrator import Orchestrator, Tier, OrchestratorResult, generate_verdict
from epistemic_agents.tracker import UsageTracker, ProviderUsage, ProviderContribution, SessionStats
from epistemic_agents.feedback import SessionFeedback, FeedbackLog, collect_feedback
from epistemic_agents.rag import build_rag_context

# New modules
from epistemic_agents.confidence import aggregate_confidence, aggregate_beliefs_confidence, extremize, aggregate_confidence_extremized
from epistemic_agents.belief_extractor import extract_beliefs, ExtractedBeliefs
from epistemic_agents.agreement_detector import detect_agreements, detect_tensions
from epistemic_agents.position_tracker import StanceShift, track_positions, format_position_summary, detect_sycophancy, compute_deltas
from epistemic_agents.context_manager import estimate_tokens, manage_context
from epistemic_agents.prediction_market import PredictionMarket, Prediction, Resolution, ProviderTrackRecord, PairwiseTracker, DomainRecord
from epistemic_agents.adversarial_graph import AttackGraph, AttackNode, build_attack_graph
from epistemic_agents.tournament import TournamentResult, TournamentLog, run_tournament
from epistemic_agents.knowledge_base import KnowledgeBase, KnowledgeEntry
from epistemic_agents.calibration_games import (
    CalibrationTask,
    CalibrationGameResult,
    CALIBRATION_TASKS,
    run_calibration_game,
)
from epistemic_agents.ledger import classify_domain
from epistemic_agents.memory import EpistemicMemory, MemoryResult
from epistemic_agents.auto_verify import classify_verifiable, auto_verify
from epistemic_agents.consensus_audit import ConsensusAuditReport, ConsensusResult, run_consensus_audit

__all__ = [
    # Core epistemic protocol
    "Belief",
    "BeliefGrounding",
    "ChallengedBelief",
    "ConfidenceLevel",
    "CONFIDENCE_LEVEL_TO_SCORE",
    "DecisionBoundary",
    "StrategicHandoff",
    "Escalation",
    "EscalationSeverity",
    "EscalationType",
    "ExecutorFeedback",
    "AmendmentType",
    "ThinkerAmendment",
    "ConversationEntry",
    "ConversationLog",
    "score_to_confidence_level",
    "VerificationMethod",
    "Verdict",
    "PanelResponse",
    "DebatePlan",
    "DebateCheckpoint",
    # Agents
    "Thinker",
    "Executor",
    "EpistemicLoop",
    # Ledger
    "BeliefLedger",
    "BeliefOutcome",
    "BeliefRecord",
    "classify_domain",
    # BIS
    "importance_scores",
    "rank_beliefs",
    "cascade_falsify",
    "detect_cycles",
    "topological_sort",
    # Multi-model panel
    "ProviderPosition",
    "AgreementPoint",
    "TensionPoint",
    "BlindSpot",
    "UniqueInsight",
    "PanelSynthesis",
    "BaseProvider",
    "ClaudeProvider",
    "OpenAICompatProvider",
    "GeminiProvider",
    "VirtualPanelist",
    "create_virtual_panelists",
    "CodeExecutorProvider",
    "get_available_providers",
    "provider_credit_status",
    # Per-call cost tracking
    "CallUsage",
    "CallCostTracker",
    "get_last_usage",
    "ModelPanel",
    "Synthesizer",
    # Orchestrator
    "Orchestrator",
    "Tier",
    "OrchestratorResult",
    "generate_verdict",
    # Tracker
    "UsageTracker",
    "ProviderUsage",
    "ProviderContribution",
    "SessionStats",
    # Feedback
    "SessionFeedback",
    "FeedbackLog",
    "collect_feedback",
    # RAG
    "build_rag_context",
    # Confidence
    "aggregate_confidence",
    "aggregate_beliefs_confidence",
    "extremize",
    "aggregate_confidence_extremized",
    # Belief extraction
    "extract_beliefs",
    "ExtractedBeliefs",
    # Agreement detection
    "detect_agreements",
    "detect_tensions",
    # Position tracking
    "StanceShift",
    "track_positions",
    "format_position_summary",
    # Context management
    "estimate_tokens",
    "manage_context",
    # Prediction market
    "PredictionMarket",
    "Prediction",
    "Resolution",
    "ProviderTrackRecord",
    "PairwiseTracker",
    "DomainRecord",
    # Adversarial graph
    "AttackGraph",
    "AttackNode",
    "build_attack_graph",
    # Tournament
    "TournamentResult",
    "TournamentLog",
    "run_tournament",
    # Knowledge base
    "KnowledgeBase",
    "KnowledgeEntry",
    # Calibration games
    "CalibrationTask",
    "CalibrationGameResult",
    "CALIBRATION_TASKS",
    "run_calibration_game",
    # Memory facade
    "EpistemicMemory",
    "MemoryResult",
    # Auto-verify
    "classify_verifiable",
    "auto_verify",
    # Consensus audit
    "ConsensusAuditReport",
    "ConsensusResult",
    "run_consensus_audit",
]
