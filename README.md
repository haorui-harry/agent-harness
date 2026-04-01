# 🤖 Agent Harness

> **Agents think. Skills act. The harness decides.** 🎯
>
> A multi-agent system where each agent dynamically selects complementary skills with full routing trace.

---

## ✨ Core Features

| 🎪 | Feature | Details |
|:---:|---------|---------|
| 🧠 | **Agent Routing** | Complexity estimation + collaboration hints |
| 🛠️ | **Skill Arsenal** | 12 built-in + 6+ marketplace skills |
| ⚡ | **Complementarity Engine V2** | Synergy detection • Conflict avoidance • Budget-aware |
| 🎭 | **Personality Engine** | Adaptive profiles + role-slot portfolio planning |
| 🎛️ | **System Modes** | `fast` ⚙️ `balanced` ⚖️ `deep` 🔍 `safety_critical` 🛡️ |
| 🔍 | **Conflict Detection** | Cross-skill consensus building + DISSENT branch |
| 📊 | **Multi-View Traces** | Sankey • Skill Matrix • Gantt • Waterfall |
| 🏷️ | **SkillCard** | Inspect built-in & external skill lifecycle |

---

## 🏗️ Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                          🎯 START                               │
├─────────────────────────────────────────────────────────────────┤
│ 1️⃣  query_understanding      → Mode + Risk + Constraints       │
│ 2️⃣  route_agent              → AURORA-style scoring            │
│ 3️⃣  adapt_personality        → Dynamic adaptation              │
│ 4️⃣  route_skills             → Complementary selection v2      │
│ 5️⃣  execute                  → Skill execution + retry         │
│ 6️⃣  detect_conflicts         → Cross-skill detection           │
│ 7️⃣  build_consensus          → Shared themes + agreement       │
│ 8️⃣  dissent                  → Structured disagreement         │
│ 9️⃣  aggregate                → Ensemble synthesis + metrics    │
├─────────────────────────────────────────────────────────────────┤
│                          🏁 END                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

Get up and running in seconds:

```bash
python -m app.main run "Summarize this report and highlight the main risks"
```

---

## 📚 Harness Engineering Layer

Located in `app/harness/` — your control center for intelligent task execution:

- 🔌 **Tool Integration** - API/browser/code adapters (`ToolRegistry`)
- ⏰ **Task Scheduling** - Planner loop (`HarnessPlanner` + `HarnessEngine.run`)
- 💾 **Memory & Context** - Persistent state (`HarnessMemoryStore`)
- 🛡️ **Guardrails** - Constraints & blocking rules (`GuardrailEngine`)
- 📈 **Evaluation Suite** - Harness metrics + eval (`HarnessEvaluator`, `harness-eval`)

---

## 🎮 CLI Commands

| Command | What It Does |
|---------|-------------|
| `run` | 🚀 End-to-end routing pipeline |
| `trace` | 🔍 Show full reasoning path |
| `benchmark` | 📊 Compare routing strategies |
| `market-search` | 🔎 Find marketplace skills |
| `personality` | 🎭 List/blend/view profiles |
| `ecosystem` | 🌍 Browse trending skills & providers |
| `mode-compare` | ⚖️ Compare across system modes |
| `skill-card` | 🏷️ Inspect skill metadata & lifecycle |
| `policy` | 📋 View mode-specific policies |
| `harness` | ⚙️ Run planner/tools/memory loop |
| `harness-eval` | 📈 Run evaluation suite |
| `replay` / `traces` | 🎬 Replay & audit past runs |

---

## 📁 Project Structure

```
agent-harness/
│
├── 🤖 app/
│   ├── agents/          8 agent profiles with adaptive personalities
│   ├── benchmark/       Dataset, strategies, evaluation framework
│   ├── coordination/    Conflict detection & consensus building
│   ├── core/            Core enums, models, graph state
│   ├── ecosystem/       Marketplace discovery & reputation system
│   ├── harness/         🎯 Planner, tools, memory, guardrails, eval
│   ├── memory/          Online learning & reliability stats
│   ├── personality/     Profiles, strategy engine, adaptation logic
│   ├── policy/          Mode-aware policy bundles
│   ├── routing/         Agent/skill routing & executor
│   ├── tracing/         Events, visualizer, analyzer
│   ├── utils/           Rich display helpers
│   ├── graph.py         8-node LangGraph topology
│   ├── main.py          CLI entry point
│   └── demo.py          Feature demonstrations
│
├── 📊 data/             Generated marketplace & benchmark datasets
├── 🧪 tests/            Comprehensive test suite
├── 📦 requirements.txt   Dependencies
└── 📖 README.md         You are here!
```

---

## 💡 Example Commands

```bash
# 🛡️ Safety-first analysis
python -m app.main run --mode safety_critical --contract "Audit this proposal"

# 🎯 Targeted skill selection  
python -m app.main run --style cautious --max-skills 4 "Evaluate proposal & list risks"

# 🔍 Full reasoning trace
python -m app.main trace --views "Analyze risks and compare options"

# 🌍 Explore marketplace
python -m app.main ecosystem trending

# 🎭 Blend personalities
python -m app.main personality --blend "scholar:0.6,explorer:0.4"

# 📊 Run full evaluation
python -m app.main harness-eval

# 🎬 Replay past runs
python -m app.main traces

# 🎨 View all demos
python -m app.main demo all
```

---

## 🔗 External Skills & Marketplace Integration

Expand your capabilities with third-party skills:

- 📦 Import marketplace skill bundles (`import_marketplace_from_file`)
- 🔌 Register runtime external skills via `app/skills/registry.py`
- 🔍 Hybrid search: Cosine similarity + BM25 + reputation scoring
- 📝 Sample bundles in `examples/external_skills.sample.json` and `examples/marketplace_bundle.sample.json`

```bash
# 📥 Load external skills
python -m app.main import-external-skills examples/external_skills.sample.json

# 📥 Import marketplace bundle
python -m app.main import-marketplace examples/marketplace_bundle.sample.json
```

---

## 🗺️ Roadmap

- 📈 Enhanced benchmark coverage & explainability
- 🔗 Real external tool backends for marketplace skills  
- 🧠 Online learning from user feedback

---

## 🎯 Why Agent Harness?

✅ **Multi-Agent Intelligence** — Leverage 8 distinct agent personalities  
✅ **Smart Skill Selection** — Complementarity engine picks the perfect combination  
✅ **Full Transparency** — Complete routing traces & reasoning paths  
✅ **Adaptive Behavior** — Personalities evolve based on context  
✅ **Safety First** — Built-in guardrails & conflict detection  
✅ **Extensible** — Bring your own skills via marketplace  

---

**Ready to harness the power of coordinated intelligence?** 🚀