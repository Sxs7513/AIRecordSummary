"""Durable, resumable streaming infrastructure shared by all LLM generations.

Import concrete contracts and services from their defining modules so importing a
lightweight contract never initializes the service dependency graph.
"""
