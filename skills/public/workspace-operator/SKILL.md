---
name: workspace-operator
description: Ground a task in the local workspace by scanning files, extracting signals, and preparing executable next actions.
category: utility
owner: agent-harness
version: 1.0.0
tags: workspace,files,repo,operator
tools: workspace_file_search,workspace_file_read,task_graph_builder
skills: codebase_triage,decompose_task,validation_planner
artifacts: workspace_brief,execution_plan
runtime_requirements: workspace
---

# Workspace Operator

Use this package when the task depends on repository or local file context, even if it is not a pure coding task.
