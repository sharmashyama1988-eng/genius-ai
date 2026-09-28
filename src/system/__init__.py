"""System Access Module with Safety Policy, Autonomous Agent Loop & Intent Routing."""

from .guard import ActionSafetyLevel, SafetyGuard
from .executor import SystemExecutor, ExecutionResult
from .workspace import WorkspaceManager
from .chat_viewer import ChatViewer
from .tools import ToolExecutor, ToolResult
from .agent_loop import AgentLoop
from .git_ops import GitOps
from .codebase_indexer import CodebaseIndexer, ProtocolRegistry
from .intent_router import IntentRouter, IntentType, RoutedIntent

__all__ = [
    "ActionSafetyLevel",
    "SafetyGuard",
    "SystemExecutor",
    "ExecutionResult",
    "WorkspaceManager",
    "ChatViewer",
    "ToolExecutor",
    "ToolResult",
    "AgentLoop",
    "GitOps",
    "CodebaseIndexer",
    "ProtocolRegistry",
    "IntentRouter",
    "IntentType",
    "RoutedIntent",
]
