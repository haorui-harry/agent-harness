<p align="right">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/Language-English-0f766e?style=for-the-badge"></a>
  <a href="./README.zh-CN.md"><img alt="Simplified Chinese" src="https://img.shields.io/badge/Language-Simplified%20Chinese-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:081826,35:0F766E,70:2563EB,100:F59E0B&height=220&section=header&text=Agent%20Harness&fontSize=54&fontColor=ffffff&desc=A%20general-purpose%20agent%20runtime%20that%20ships%20real%20deliverables.&descAlignY=68&animation=fadeIn" alt="Agent Harness banner"/>
</p>

<p align="center">
  <b>Agent Harness is a thread-first general agent runtime: it plans with capabilities, uses skills and tools when needed, works inside a real workspace, and leaves behind inspectable deliverables instead of only an answer string.</b>
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

- a research report
- a patch draft
- an engineering handoff memo
- a benchmark manifest or run config
- a launch memo or rollout pack
- a slide deck plan or webpage blueprint
- a delivery bundle with linked artifacts

Supporting artifacts can include:

- evidence bundles
- workspace findings
- validation plans or execution traces
- source matrices
- interoperability exports for external skill ecosystems

## Core Model

`Request -> Thread -> Capability Planner -> Skills/Tools/Workspace Actions -> Main Deliverable -> Evidence + Artifact Bundle`

This is deliberately simpler than a task-specific workflow catalog.

The runtime should decide how to use skills, tools, and workspace actions based on the task, not because the task was shoved into a hard-coded funnel.

## What Makes It Different

### 1. Deliverable-first

The first-class result is the deliverable the user asked for.

Not the scorecard.
Not the bundle.
Not the planner trace.

Those still exist, but they support the main result instead of replacing it.

### 2. Generic By Default

The runtime is not supposed to be “the research flow” or “the code flow”.

It starts from a task spec, available capabilities, current evidence gaps, and workspace state. That keeps the system usable across code, research, operations, and mixed tasks.

### 3. Thread + Workspace + Recovery

Each task runs inside a persistent thread with:

- resumable execution
- retry / interrupt / recovery
- workspace artifacts
- event stream / snapshot export

### 4. Skills Without Lock-In

Skills are packaged capabilities, not the entire product. The runtime can also export an interoperability catalog so external OpenAI/Anthropic-style ecosystems can consume the project’s capabilities.

---

## Demo Gallery

Tracked demo snapshots live in `docs/demo/`.

Open the full index: [docs/demo/README.md](./docs/demo/README.md)

### Live Fintech Launch

- Deliverable: [docs/demo/live/deliverable.md](./docs/demo/live/deliverable.md)
- Showcase: [docs/demo/live/showcase.html](./docs/demo/live/showcase.html)
- Press Brief: [docs/demo/live/press-brief.md](./docs/demo/live/press-brief.md)

### Enterprise Rollout Pack

- Deliverable: [docs/demo/enterprise/deliverable.md](./docs/demo/enterprise/deliverable.md)
- Showcase: [docs/demo/enterprise/showcase.html](./docs/demo/enterprise/showcase.html)
- Press Brief: [docs/demo/enterprise/press-brief.md](./docs/demo/enterprise/press-brief.md)

### Research Promotion Pack

- Deliverable: [docs/demo/research/deliverable.md](./docs/demo/research/deliverable.md)
- Showcase: [docs/demo/research/showcase.html](./docs/demo/research/showcase.html)
- Press Brief: [docs/demo/research/press-brief.md](./docs/demo/research/press-brief.md)

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Create A Persistent Thread

```bash
python -m app.main agent-thread-create "General Agent Demo"
```

### 3. Run A Generic Task

```bash
python -m app.main agent-thread-run <thread_id> "Produce a deep research report on agent runtime reliability" --target auto
```

### 4. Inspect The Thread Workspace

```bash
python -m app.main agent-thread-workspace-view <thread_id> --output reports/thread_view
```

### 5. Run The Studio Showcase

```bash
python -m app.main studio-showcase "Design a launch plan for a regulated AI copilot" --tag demo
```

### 6. Enable A Live Model

Use environment variables or CLI flags. Keep secrets outside the repo.

```bash
set AGENT_HARNESS_MODEL_BASE_URL=https://your-endpoint/v1
set AGENT_HARNESS_MODEL_API_KEY=your_api_key
set AGENT_HARNESS_MODEL_NAME=your_model
python -m app.main harness-live "Write a benchmark-backed research brief"
```

### 7. Run Tests

```bash
pytest -q
```

---

## Main Commands

| Goal | Command |
|---|---|
| Create thread | `python -m app.main agent-thread-create "My Task"` |
| List threads | `python -m app.main agent-threads` |
| Run generic super-agent | `python -m app.main agent-thread-run <thread_id> "<query>" --target auto` |
| Execute task graph directly | `python -m app.main agent-thread-exec-task <thread_id> "<query>" --target general` |
| Export workspace view | `python -m app.main agent-thread-workspace-view <thread_id>` |
| Export thread snapshot | `python -m app.main agent-thread-export <thread_id>` |
| Run harness with live model | `python -m app.main harness-live "<query>"` |
| Build code mission pack | `python -m app.main harness-code-pack "<query>" --workspace .` |
| Run benchmark suite | `python -m app.main benchmark-suite --adapters routing_internal,lab_daily` |
| Run showcase | `python -m app.main studio-showcase "<query>" --tag demo` |
| Export skills interop | `python -m app.main skills-interop-export` |

---

## Repository Layout

### Runtime

- `app/harness/`: planner, live orchestration, evaluation, reporting, visuals
- `app/agents/`: thread runtime, workspace action mapper, scheduler, sandbox, workspace view
- `app/skills/`: built-in skills, packages, interop export
- `app/core/`: task spec, capability graph, contracts, shared state

### Product Surfaces

- `app/studio/`: showcase and release-facing packaging
- `docs/demo/`: tracked demo snapshots that ship with the repository
- `reports/`: runtime outputs generated locally

### Validation

- `tests/`: runtime, showcase, planner, and live orchestration tests

---

## Honest Status

Agent Harness is strongest today when a task benefits from:

- a persistent thread
- artifact-producing execution
- evidence-aware synthesis
- inspectable outputs

It is not finished.

The main direction is to keep simplifying the generic runtime so it relies less on task-family templates and more on reusable capability planning plus strong deliverables.
