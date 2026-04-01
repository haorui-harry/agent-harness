# Agent Harness

Agents think. Skills act. The harness decides.

A multi-agent system where each agent dynamically selects complementary skills with full routing trace.

## Highlights

- Agent routing with complexity estimation and collaboration hints
- 12 built-in skills + marketplace discovery (6+ external skills)
- Complementarity Engine V2 (synergy, conflict avoidance, budget-aware selection, refinement)
- Role-slot portfolio planning (evidence/reasoning/communication/verification)
- Personality engine (profiles + adaptation)
- System modes + policy center (`fast`, `balanced`, `deep`, `safety_critical`)
- Conflict detection and consensus building
- Structured DISSENT branch for high-risk/low-confidence runs
- Multi-view trace visualization (sankey path, skill matrix, execution gantt, confidence waterfall)
- SkillCard + lifecycle inspection (`skill-card`) for built-in and external skills
- Structured response contract for user/debug/evaluation integration

## Harness Layer

The repository includes a dedicated harness engineering layer in `app/harness/`:

- Tool use: API/browser/code adapters (`ToolRegistry`)
- Task scheduling: planner loop (`HarnessPlanner` + `HarnessEngine.run`)
- State management: persistent memory/context (`HarnessMemoryStore`)
- Guardrails: constraints and blocking rules (`GuardrailEngine`)
- Eval: harness metrics and evaluation suite (`HarnessEvaluator`, `harness-eval`)

## Architecture

```text
[START]
  |
query_understanding  (mode + risk + initial constraints)
  |
route_agent          (AURORA-style scoring + complexity + collaboration)
  |
adapt_personality    (dynamic personality adaptation)
  |
route_skills         (complementary selection v2)
  |
execute              (skill execution + retry + quality)
  |
detect_conflicts     (cross-skill conflict detection)
  |
build_consensus      (shared themes + agreement strength)
  |
dissent              (structured disagreement branch)
  |
aggregate            (ensemble synthesis + metrics + contract)
  |
[END]
```

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/<your-name>/langgraph-skill-router.git
cd langgraph-skill-router
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux (bash)

```bash
git clone https://github.com/<your-name>/langgraph-skill-router.git
cd langgraph-skill-router
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
python -m app.main run "Summarize this report and highlight the main risks"
```

## CLI Reference

- `run` - run end-to-end routing pipeline
- `benchmark` - run benchmark comparison
- `market-search` - search marketplace skills
- `rate-skill` - submit rating for marketplace skill
- `personality` - list/view/blend personality profiles
- `trace` - show reasoning path for a query
- `ecosystem` - browse/trending/providers/tags for marketplace
- `analyze` - run routing quality analysis
- `demo` - run feature demos
- `policy` - inspect policy bundle for a system mode
- `mode-compare` - compare routing decisions across system modes
- `replay` / `traces` - replay and audit persisted trace runs
- `import-marketplace` - import third-party marketplace bundle from JSON
- `import-external-skills` - load runtime third-party skills from JSON
- `skill-card` - inspect a skill's metadata + lifecycle status
- `harness` - run planner/tools/memory/guardrails loop
- `harness-eval` - run harness-level evaluation suite

## Project Structure

```text
langgraph-skill-router/
├── app/
│   ├── agents/          # 8 agent profiles with personality and collaboration preferences
│   ├── benchmark/       # benchmark dataset/strategies/evaluation
│   ├── coordination/    # conflict detection, resolution, consensus
│   ├── core/            # enums, models, graph state
│   ├── ecosystem/       # marketplace models/search/reputation/store
│   ├── harness/         # planner/tools/memory/guardrails/eval layer
│   ├── memory/          # online learning counters and reliability stats
│   ├── personality/     # profiles, strategy engine, adaptation
│   ├── policy/          # mode-aware policy bundles
│   ├── routing/         # agent router, skill router, complementarity, executor
│   ├── tracing/         # tracing events, visualizer, analyzer
│   ├── utils/           # rich display helpers
│   ├── graph.py         # 8-node LangGraph topology
│   ├── main.py          # CLI entrypoint
│   └── demo.py          # demos
├── data/                # generated marketplace and benchmark data
├── tests/               # test suite
├── requirements.txt
└── README.md
```

## Example Commands

```bash
python -m app.main run --style cautious --max-skills 4 "Evaluate this proposal and list key risks"
python -m app.main run --mode safety_critical --contract "Audit this recommendation and challenge weak points"
python -m app.main trace --views "Analyze risks and compare options and recommend a plan"
python -m app.main mode-compare "Audit this plan and propose safe mitigations"
python -m app.main traces
python -m app.main ecosystem trending
python -m app.main personality --blend "scholar:0.6,explorer:0.4"
python -m app.main import-external-skills examples/external_skills.sample.json
python -m app.main import-marketplace examples/marketplace_bundle.sample.json
python -m app.main skill-card risk_heatmap
python -m app.main harness "Audit this proposal and challenge assumptions"
python -m app.main harness-eval
python -m app.main demo all
```

## External Skill & Resource Extension

- Marketplace supports importing third-party skill bundles (`import_marketplace_from_file` in `app/ecosystem/store.py`)
- Runtime external skill registration is available in `app/skills/registry.py`
- Hybrid search (`cosine + BM25 + reputation`) enables integrating external skill catalogs
- Sample external resource bundles are provided in `examples/external_skills.sample.json` and `examples/marketplace_bundle.sample.json`

## Roadmap

- Better benchmark coverage and strategy explainability
- Real external tool backends for marketplace skills
- Online learning for routing weights from feedback

## License

MIT

## Citation

```bibtex
@software{langgraph_skill_router,
  title = {LangGraph Skill Router},
  year = {2026},
  note = {Dynamic multi-agent skill routing with complementarity and traceability}
}
```
