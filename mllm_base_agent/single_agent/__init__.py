"""Single-agent (one embodiment) variants of SpatialWorld environments.

Mirrors the layout of :mod:`mllm_base_agent.dual_agent`, but for the
standard single-body agent loop driven by ``mllm_base_agent.agent.runner``.
Currently this package only hosts the per-environment skill-memory
libraries (e.g. ``single_agent/ai2thor/core/memory/``); the actual agent
loop lives in ``mllm_base_agent/agent/`` and is shared across environments.
"""
