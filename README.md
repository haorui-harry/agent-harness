<p align="right">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/Language-English-0f766e?style=for-the-badge"></a>
  <a href="./README.zh-CN.md"><img alt="Simplified Chinese" src="https://img.shields.io/badge/Language-Simplified%20Chinese-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:081826,35:0F766E,70:2563EB,100:F59E0B&height=220&section=header&text=Agent%20Harness&fontSize=54&fontColor=ffffff&desc=A%20general-purpose%20agent%20runtime%20that%20ships%20real%20deliverables.&descAlignY=68&animation=fadeIn" alt="Agent Harness banner"/>
</p>

<p align="center">
  <b>A high-performance, general-purpose agent runtime: plans with capabilities, uses skills and tools on demand, runs inside a real workspace, and ships inspectable deliverables -- not just answer strings.</b>
</p>

---

## One Line

Agent Harness turns an open-ended request into a main deliverable plus the evidence, artifacts, and runtime trace needed to review or continue the work.

## Why This Project

The best general agents are usually not the most complicated ones.

What matters is a small number of strong primitives:

- one persistent thread
- one planner that reasons over capabilities instead of fixed flows
- one workspace where actions can create real files
- one main deliverable that the user can actually open
- one artifact/evidence rail for audit, retry, and follow-up

That is the design center of Agent Harness.

## What It Can Produce

Depending on the task, the main deliverable can be:

- a research report with evidence-backed findings
- an architecture design with trade-off analysis
- a comparison matrix with concrete recommendations
- a patch draft or engineering handoff memo
- a slide deck plan or webpage blueprint
- a delivery bundle with linked artifacts

Supporting artifacts can include:

- evidence bundles (live search + static references)
- workspace findings
- validation plans or execution traces
- source matrices
- interoperability exports for external skill ecosystems

## Core Model

```
Request -> Thread -> Capability Planner -> Skills/Tools/Workspace Actions -> Main Deliverable -> Evidence + Artifact Bundle
```

This is deliberately simpler than a task-specific workflow catalog.

The runtime decides how to use skills, tools, and workspace actions based on the task -- not because the task was shoved into a hard-coded funnel.

## Agent Loop

The runtime is built around one short loop:

1. open or resume one thread
2. infer the main deliverable and missing channels from the task
3. inspect skills, tools, web context, or workspace only when the task calls for them
4. execute a small task graph inside the thread workspace
5. publish one primary deliverable plus reviewable evidence and follow-up artifacts

## Live Agent Enhancement

When a live model API is configured, the runtime adds a **4-stage reasoning chain**:

1. **Analysis** -- decompose the task, map available evidence, identify gaps
2. **Synthesis** -- produce a structured answer grounded in analysis findings
3. **Critique** -- peer-review the synthesis: flag red flags, blind spots, unsupported claims
4. **Revision** -- fix every issue the critique identified, with mandatory fix targets injected into the prompt

This chain means the final output has been through analysis, writing, review, and revision -- not just one-shot generation.

---

## Demo Gallery

All demos are generated with real API calls (GPT-4o via live agent). Each demo runs the full agent loop: planning, evidence collection, 4-stage live reasoning, and deliverable packaging.

### Research: Reducing LLM Hallucination

> "What are the top 3 most impactful techniques for reducing LLM hallucination in production systems?"

- [Full output](./docs/demo/research-hallucination.md)
- 12 evidence records | 4 live agent calls | Value index 74.17
- Covers: RAG, constrained decoding, fine-tuning with RLHF

### Engineering: Rate Limiting Architecture

> "Design a rate limiting system for a high-traffic REST API. Include architecture, algorithms, and failure handling."

- [Full output](./docs/demo/engineering-ratelimit.md)
- 6 evidence records | 4 live agent calls | Value index 75.37
- Covers: Token Bucket vs Sliding Window, Redis vs DynamoDB state stores, circuit breakers, monitoring

### Analysis: Database Comparison for SaaS

> "Compare PostgreSQL, MongoDB, and DynamoDB for a multi-tenant SaaS app handling 10k req/s."

- [Full output](./docs/demo/analysis-database.md)
- 6 evidence records | 4 live agent calls | Value index 78.04
- Covers: comparison matrix, scaling trade-offs, concrete recommendation with justification

---

## What Makes It Different

### 1. Deliverable-first

The first-class result is the deliverable the user asked for -- not the scorecard, not the bundle, not the planner trace.

### 2. Generic By Default

The runtime is not "the research flow" or "the code flow". It starts from a task spec, available capabilities, current evidence gaps, and workspace state. That keeps the system usable across code, research, operations, and mixed tasks.

### 3. Critique-Driven Quality

The live agent doesn't just generate -- it reviews its own output. Critique findings (red flags, blind spots, improvement items) are injected as **mandatory fix targets** in the revision pass. This is measurably better than one-shot generation.

### 4. Real Evidence, Not Templates

Evidence collection uses live search via the LLM API when configured, producing task-specific references instead of hardcoded static links. The evidence digest flows directly into synthesis and revision prompts.

### 5. Thread + Workspace + Recovery

Each task runs inside a persistent thread with resumable execution, retry/interrupt/recovery, workspace artifacts, and event stream export.

### 6. Skills Without Lock-In

Skills are packaged capabilities, not the entire product. The runtime can export an interoperability catalog so external OpenAI/Anthropic-style ecosystems can consume the project's capabilities.

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run A Task (No API Key Required)

```python
from app.harness.engine import HarnessEngine
from app.harness import HarnessConstraints

engine = HarnessEngine()
run = engine.run(
    query="Summarize the key risks of deploying LLMs in production",
    constraints=HarnessConstraints(max_steps=5, max_tool_calls=4),
)
print(run.final_answer[:2000])
```

### 3. Run With Live Model (Higher Quality)

```python
from app.harness.engine import HarnessEngine
from app.harness import HarnessConstraints

engine = HarnessEngine()
run = engine.run(
    query="Design a caching strategy for a microservices architecture",
    constraints=HarnessConstraints(
        max_steps=5,
        max_tool_calls=4,
        enable_live_agent=True,
        max_live_agent_calls=4,
    ),
    live_model={
        "base_url": "https://your-endpoint/v1",
        "api_key": "your_api_key",
        "model_name": "gpt-4o",
    },
)
print(run.final_answer[:3000])
```

Or use environment variables:

```bash
export AGENT_HARNESS_MODEL_BASE_URL=https://your-endpoint/v1
export AGENT_HARNESS_MODEL_API_KEY=your_api_key
export AGENT_HARNESS_MODEL_NAME=gpt-4o
python -m app.main harness-live "Write an evidence-backed research brief"
```

### 4. Run Tests

```bash
pytest -q
# 183 tests passing
```

---

## Main Commands

| Goal | Command |
|---|---|
| Create thread | `python -m app.main agent-thread-create "My Task"` |
| List threads | `python -m app.main agent-threads` |
| Run generic super-agent | `python -m app.main agent-thread-run <thread_id> "<query>" --target auto` |
| Execute task graph | `python -m app.main agent-thread-exec-task <thread_id> "<query>" --target general` |
| Export workspace view | `python -m app.main agent-thread-workspace-view <thread_id>` |
| Run harness with live model | `python -m app.main harness-live "<query>"` |
| Build code mission pack | `python -m app.main harness-code-pack "<query>" --workspace .` |
| Run showcase | `python -m app.main studio-showcase "<query>" --tag demo` |
| Export skills interop | `python -m app.main skills-interop-export` |

---

## Repository Layout

### Runtime

- `app/harness/`: planner, live orchestration, evidence, evaluation, reporting
- `app/agents/`: thread runtime, workspace action mapper, scheduler, sandbox
- `app/skills/`: built-in skills, packages, interop export
- `app/core/`: task spec, capability graph, contracts, shared state

### Product Surfaces

- `app/studio/`: showcase and release packaging
- `docs/demo/`: live demo outputs generated with real API calls
- `reports/`: runtime outputs generated locally

### Validation

- `tests/`: 183 tests covering runtime, showcase, planner, and live orchestration

---

## Architecture

```
                    +------------------+
                    |   User Request   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Capability Planner|  (configurable limits, language-agnostic)
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     | Skill Router|  | Tool Engine |  |  Workspace  |
     | (26 skills) |  | (15+ tools) |  | (file I/O)  |
     +--------+---+  +------+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |  Evidence Bundle  |  (live search + static catalog)
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Live Agent     |  (analysis -> synthesis -> critique -> revision)
                    +--------+---------+
                             |
                    +--------v---------+
                    | Main Deliverable |  (task-adaptive format)
                    +------------------+
```

---

## Honest Status

Agent Harness is strongest today when a task benefits from:

- evidence-aware synthesis (not just generation)
- structured multi-pass reasoning (analysis + critique + revision)
- persistent thread with inspectable artifacts
- task-adaptive output format (not forced templates)

What still matters next:

- connect real web search APIs for live evidence collection
- strengthen long-horizon thread execution for complex multi-step tasks
- improve evidence grounding so every claim traces to a named source
- reduce the gap between live-agent-enhanced and base-only output quality
