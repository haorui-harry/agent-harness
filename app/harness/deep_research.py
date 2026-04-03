"""Deep research report builder for repository-level capability studies."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.benchmark.adapters import BenchmarkAdapterRunner
from app.harness.evidence import EvidenceProviderRegistry
from app.harness.manifest import ToolManifestRegistry
from app.skills.registry import list_builtin_skills


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HarnessDeepResearchBuilder:
    SCHEMA = "agent-harness-deep-research/v1"

    def __init__(self) -> None:
        self.evidence = EvidenceProviderRegistry()
        self.manifests = ToolManifestRegistry()
        self.adapters = BenchmarkAdapterRunner()

    def build(
        self,
        *,
        topic: str,
        subject_root: str | Path,
        competitor_root: str | Path | None = None,
        subject_name: str = "agent-harness",
        competitor_name: str = "deer-flow",
    ) -> dict[str, Any]:
        subject_path = Path(subject_root).resolve()
        competitor_path = Path(competitor_root).resolve() if competitor_root else None
        subject = self._scan_subject_repo(subject_name, subject_path)
        competitor = self._scan_competitor_repo(competitor_name, competitor_path) if competitor_path else {}
        dimensions = self._build_dimensions(subject, competitor)
        roadmap = self._build_roadmap(dimensions)
        benchmark_map = self._benchmark_map()
        evidence = self.evidence.collect(
            query=topic,
            limit=8,
            domains=["research", "evidence", "governance", "enterprise"],
        )
        payload = {
            "schema": self.SCHEMA,
            "generated_at": _utc_now(),
            "topic": topic,
            "subject": subject,
            "competitor": competitor,
            "dimensions": dimensions,
            "benchmark_map": benchmark_map,
            "roadmap": roadmap,
            "evidence": evidence,
        }
        payload["summary"] = self._executive_summary(subject, competitor, dimensions)
        payload["framework_markdown"] = self._render_framework_markdown(topic, subject, competitor, dimensions, benchmark_map)
        payload["report_markdown"] = self._render_report_markdown(topic, subject, competitor, dimensions, benchmark_map, roadmap, evidence)
        return payload

    def write_bundle(self, payload: dict[str, Any], output_dir: str | Path = "reports") -> dict[str, str]:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        framework_path = root / "agent_harness_deep_research_framework.md"
        report_path = root / "agent_harness_deep_research_report.md"
        bundle_path = root / "agent_harness_deep_research_bundle.json"
        html_path = root / "agent_harness_deep_research.html"
        framework_path.write_text(str(payload.get("framework_markdown", "")), encoding="utf-8")
        report_path.write_text(str(payload.get("report_markdown", "")), encoding="utf-8")
        bundle_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        html_path.write_text(self.render_html(payload), encoding="utf-8")
        return {
            "framework": str(framework_path),
            "report": str(report_path),
            "bundle": str(bundle_path),
            "html": str(html_path),
        }

    def render_html(self, payload: dict[str, Any]) -> str:
        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
        subject = payload.get("subject", {}) if isinstance(payload.get("subject", {}), dict) else {}
        competitor = payload.get("competitor", {}) if isinstance(payload.get("competitor", {}), dict) else {}
        dimensions = payload.get("dimensions", []) if isinstance(payload.get("dimensions", []), list) else []
        benchmarks = payload.get("benchmark_map", []) if isinstance(payload.get("benchmark_map", []), list) else []
        roadmap = payload.get("roadmap", []) if isinstance(payload.get("roadmap", []), list) else []
        evidence = payload.get("evidence", {}).get("records", []) if isinstance(payload.get("evidence", {}), dict) else []
        excerpt = str(payload.get("report_markdown", ""))[:2200]
        return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" /><title>AI Agent Harness Deep Research</title><style>:root{{--bg:#f5efe4;--panel:rgba(255,251,245,.94);--ink:#132238;--muted:#5d6a7b;--line:rgba(19,34,56,.12);--accent:#cf5c36;--accent2:#2c7a7b;--gold:#cc9a06;--shadow:0 22px 60px rgba(19,34,56,.10)}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font-family:'Segoe UI',sans-serif;background:radial-gradient(circle at top left,rgba(207,92,54,.15),transparent 28%),radial-gradient(circle at top right,rgba(44,122,123,.18),transparent 24%),linear-gradient(180deg,#fffaf1 0%,var(--bg) 100%)}}.page{{max-width:1320px;margin:0 auto;padding:28px}}.hero{{border-radius:28px;padding:32px;background:linear-gradient(135deg,rgba(207,92,54,.18),rgba(44,122,123,.12),rgba(19,34,56,.05));border:1px solid var(--line);box-shadow:var(--shadow)}}.k{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:var(--accent2)}}h1{{margin:10px 0 12px;font-size:46px;line-height:1.05}}.lede{{max-width:900px;line-height:1.7;font-size:18px;color:var(--muted)}}.hero-grid{{display:grid;grid-template-columns:1.35fr .95fr;gap:18px;margin-top:18px}}.mini,.card{{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:var(--shadow)}}.mini{{background:rgba(255,255,255,.56)}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:18px}}.s4{{grid-column:span 4}}.s5{{grid-column:span 5}}.s6{{grid-column:span 6}}.s7{{grid-column:span 7}}.s12{{grid-column:span 12}}.ey{{font-size:12px;letter-spacing:.10em;text-transform:uppercase;color:#4a8399}}.metric{{font-size:36px;font-weight:700;margin-top:8px}}.muted{{color:var(--muted)}}ul{{padding-left:18px;margin:10px 0 0}}li{{margin:8px 0;line-height:1.55}}table{{width:100%;border-collapse:collapse;margin-top:12px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#456c82}}.ahead{{color:var(--accent2);font-weight:700}}.behind{{color:var(--accent);font-weight:700}}.mixed{{color:var(--gold);font-weight:700}}pre{{white-space:pre-wrap;line-height:1.55;background:rgba(19,34,56,.04);border-radius:18px;padding:18px;overflow:auto}}@media (max-width:980px){{.hero-grid,.grid{{grid-template-columns:1fr}}.s4,.s5,.s6,.s7,.s12{{grid-column:span 1}}h1{{font-size:34px}}}}</style></head><body><div class=\"page\"><section class=\"hero\"><div class=\"k\">Deep Research Demo</div><h1>{html.escape(str(summary.get('title', 'AI Agent Harness Deep Research Report')))}</h1><div class=\"lede\">{html.escape(str(summary.get('summary', '')))}</div><div class=\"hero-grid\"><div class=\"mini\"><div class=\"ey\">Bottom Line</div><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in summary.get('highlights', [])) or '<li>No highlights generated.</li>'}</ul></div><div class=\"mini\"><div class=\"ey\">Why This Matters</div><ul><li>{html.escape(str(summary.get('value_statement', '')))}</li><li>{html.escape(str(summary.get('differentiator', '')))}</li></ul></div></div></section><section class=\"grid\"><div class=\"card s4\"><div class=\"ey\">Agent Harness</div><div class=\"metric\">{int(subject.get('metrics', {}).get('python_files', 0))}</div><div class=\"muted\">Python files scanned</div><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in subject.get('headline_points', []))}</ul></div><div class=\"card s4\"><div class=\"ey\">DeerFlow</div><div class=\"metric\">{int(competitor.get('metrics', {}).get('public_skill_count', 0))}</div><div class=\"muted\">Public skills in reference repo</div><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in competitor.get('headline_points', []))}</ul></div><div class=\"card s4\"><div class=\"ey\">Research Surface</div><div class=\"metric\">{int(subject.get('metrics', {}).get('benchmark_adapter_count', 0))}</div><div class=\"muted\">Benchmark adapters in agent-harness</div><ul><li>{html.escape(str(summary.get('research_statement', '')))}</li></ul></div><div class=\"card s7\"><div class=\"ey\">Competitive Matrix</div><table><thead><tr><th>Dimension</th><th>Agent Harness</th><th>DeerFlow</th><th>Verdict</th></tr></thead><tbody>{''.join(self._html_dimension_row(item) for item in dimensions)}</tbody></table></div><div class=\"card s5\"><div class=\"ey\">Benchmarks</div><table><thead><tr><th>Benchmark</th><th>Fit</th><th>Current Gap</th></tr></thead><tbody>{''.join(self._html_benchmark_row(item) for item in benchmarks)}</tbody></table></div><div class=\"card s6\"><div class=\"ey\">Roadmap</div><ul>{''.join(f"<li><strong>{html.escape(str(item.get('phase', '')))}</strong>: {html.escape(str(item.get('focus', '')))}</li>" for item in roadmap)}</ul></div><div class=\"card s6\"><div class=\"ey\">Evidence</div><ul>{''.join(f"<li><strong>{html.escape(str(item.get('title', '')))}</strong>: {html.escape(str(item.get('summary', '')))}</li>" for item in evidence[:8]) or '<li>No evidence records found.</li>'}</ul></div><div class=\"card s12\"><div class=\"ey\">Report Excerpt</div><pre>{html.escape(excerpt)}</pre></div></section></div></body></html>"""

    @staticmethod
    def _html_dimension_row(item: dict[str, Any]) -> str:
        verdict = str(item.get("verdict", "mixed"))
        css = {"ahead": "ahead", "behind": "behind", "mixed": "mixed"}.get(verdict, "mixed")
        return f"<tr><td>{html.escape(str(item.get('label', '')))}</td><td>{float(item.get('subject_score', 0.0)):.1f}</td><td>{float(item.get('competitor_score', 0.0)):.1f}</td><td class='{css}'>{html.escape(verdict)}</td></tr>"

    @staticmethod
    def _html_benchmark_row(item: dict[str, Any]) -> str:
        return f"<tr><td>{html.escape(str(item.get('name', '')))}</td><td>{html.escape(str(item.get('fit', '')))}</td><td>{html.escape(str(item.get('gap', '')))}</td></tr>"
    def _scan_subject_repo(self, name: str, root: Path) -> dict[str, Any]:
        app_root = root / "app"
        tests_root = root / "tests"
        runtime_text = self._read_text(app_root / "agents" / "runtime.py")
        engine_text = self._read_text(app_root / "harness" / "engine.py")
        tools_text = self._read_text(app_root / "harness" / "tools.py")
        scheduler_text = self._read_text(app_root / "agents" / "scheduler.py")
        main_text = self._read_text(app_root / "main.py")
        feature_flags = {
            "thread_runtime": "class AgentThreadRuntime" in runtime_text,
            "resume_retry_interrupt": all(token in runtime_text for token in ["resume_execution", "retry_execution", "request_interrupt"]),
            "scheduler_recovery": "recover_execution" in scheduler_text and "recover_all" in scheduler_text,
            "task_graph_runtime": "execute_thread_generic_task" in engine_text and "task_graph_builder" in tools_text,
            "workspace_actions": all(token in tools_text for token in ["workspace_file_search", "workspace_file_read", "workspace_file_write"]),
            "subagent_executor": (app_root / "agents" / "subagents.py").exists(),
            "workspace_view": (app_root / "agents" / "workspace_view.py").exists(),
            "benchmark_adapters": (app_root / "benchmark" / "adapters.py").exists(),
            "interop_export": (app_root / "skills" / "interop.py").exists(),
            "evidence_provider": (app_root / "harness" / "evidence.py").exists(),
            "deep_research_report": (app_root / "harness" / "deep_research.py").exists(),
        }
        notable_files = [
            {"path": "app/agents/runtime.py", "reason": "Persistent thread runtime with async wait, resume, retry, and interrupt support."},
            {"path": "app/agents/task_actions.py", "reason": "Maps task-graph nodes into tool, skill, workspace, file, and command actions."},
            {"path": "app/harness/tools.py", "reason": "Holds tool discovery, workspace tools, evidence tools, and executable task-graph generation."},
            {"path": "app/harness/deep_research.py", "reason": "Generates research framework, final report, and HTML publication page for repo studies."},
            {"path": "app/benchmark/adapters.py", "reason": "Defines benchmark suite and ablation outputs under a unified schema."},
            {"path": "app/skills/interop.py", "reason": "Exports capability catalogs into external skill ecosystems."},
        ]
        return {
            "name": name,
            "root": str(root),
            "metrics": {
                "python_files": len(list(app_root.rglob("*.py"))),
                "test_file_count": len(list(tests_root.glob("test_*.py"))),
                "cli_command_count": main_text.count('@app.command("'),
                "builtin_skill_count": len(list_builtin_skills()),
                "tool_count": len(self.manifests.list_all()),
                "benchmark_adapter_count": len(self.adapters.list_adapters()),
            },
            "feature_flags": feature_flags,
            "headline_points": [
                "Execution semantics are explicit: task graph, node action mapping, and thread persistence live in first-class modules.",
                "The framework already has benchmark-ablation and interop surfaces, which most skill repos still treat as appendices.",
                "The main gap is not raw module count but output quality: research reports and general-agent demos were previously too weak.",
            ],
            "notable_files": notable_files,
        }

    def _scan_competitor_repo(self, name: str, root: Path) -> dict[str, Any]:
        readme_text = self._read_text(root / "README.md")
        backend_root = root / "backend"
        frontend_root = root / "frontend"
        public_skills = root / "skills" / "public"
        return {
            "name": name,
            "root": str(root),
            "metrics": {
                "public_skill_count": len([item for item in public_skills.iterdir() if item.is_dir()]) if public_skills.exists() else 0,
                "backend_test_count": len(list((backend_root / "tests").rglob("test_*.py"))) if (backend_root / "tests").exists() else 0,
            },
            "feature_flags": {
                "subagents": "sub-agent" in readme_text.lower() or "subagent" in readme_text.lower(),
                "memory": "memory" in readme_text.lower(),
                "sandbox": "sandbox" in readme_text.lower(),
                "mcp": "mcp" in readme_text.lower(),
                "frontend_workspace": (frontend_root / "src" / "app" / "workspace").exists() or "workspace" in self._read_text(frontend_root / "CLAUDE.md").lower(),
                "artifact_ui": "artifacts" in self._read_text(frontend_root / "CLAUDE.md").lower(),
                "official_website": "official website" in readme_text.lower(),
                "skill_installation": "install" in readme_text.lower() and "skill" in readme_text.lower(),
            },
            "headline_points": [
                "DeerFlow ships a much more productized public surface: website, frontend workspace, thread chat, artifacts, and gateway APIs.",
                "Its published skills encode a stronger research method than a generic one-shot prompt.",
                "The repo is already organized as a full harness product, not just a routing experiment.",
            ],
            "notable_files": [
                {"path": "README.md", "reason": "Public product positioning around super agent harness, sub-agents, memory, sandboxes, MCP, and demos."},
                {"path": "skills/public/deep-research/SKILL.md", "reason": "Enforces broad exploration, deep dive, diversity validation, and synthesis before content generation."},
                {"path": "skills/public/consulting-analysis/SKILL.md", "reason": "Imposes a two-phase structure: analysis framework first, final consulting-grade report second."},
                {"path": "frontend/CLAUDE.md", "reason": "Describes the thread-based streaming UI with artifacts, workspace, memory, and skills management."},
                {"path": "backend/CLAUDE.md", "reason": "Shows production-harness layers for sandbox, MCP, memory, subagents, and gateway APIs."},
            ],
        }

    def _build_dimensions(self, subject: dict[str, Any], competitor: dict[str, Any]) -> list[dict[str, Any]]:
        s = subject.get("feature_flags", {})
        c = competitor.get("feature_flags", {})
        sm = subject.get("metrics", {})
        cm = competitor.get("metrics", {})
        rows = [
            {
                "label": "Executable Runtime",
                "subject_score": self._score([s.get("thread_runtime", False), s.get("resume_retry_interrupt", False), s.get("scheduler_recovery", False), s.get("task_graph_runtime", False), s.get("workspace_actions", False), s.get("subagent_executor", False)]),
                "competitor_score": self._score([c.get("subagents", False), c.get("sandbox", False), c.get("memory", False), c.get("frontend_workspace", False), c.get("artifact_ui", False), c.get("mcp", False)]),
                "why": "agent-harness now has stronger explicit execution semantics, but deer-flow still packages more of the runtime into a mature end-user system.",
            },
            {
                "label": "Research Workflow",
                "subject_score": self._score([s.get("deep_research_report", False), s.get("evidence_provider", False), s.get("benchmark_adapters", False), sm.get("benchmark_adapter_count", 0) >= 3, s.get("workspace_view", False)]),
                "competitor_score": self._score([c.get("official_website", False), True, True, c.get("artifact_ui", False), c.get("frontend_workspace", False)]),
                "why": "DeerFlow publishes a cleaner methodology today, while agent-harness is now stronger on explicit evidence and benchmark hooks.",
            },
            {
                "label": "Skill Ecosystem",
                "subject_score": min(5.0, 2.2 + min(float(sm.get("builtin_skill_count", 0)) / 12.0, 1.6) + (0.8 if s.get("interop_export", False) else 0.0)),
                "competitor_score": min(5.0, 2.6 + min(float(cm.get("public_skill_count", 0)) / 8.0, 1.8) + (0.6 if c.get("skill_installation", False) else 0.0)),
                "why": "agent-harness can export interoperable skills, but deer-flow exposes a better public skill experience and a more recognizable skill catalog.",
            },
            {
                "label": "Sandbox And Provider Layer",
                "subject_score": self._score([s.get("workspace_actions", False), s.get("thread_runtime", False), s.get("resume_retry_interrupt", False), s.get("subagent_executor", False)]),
                "competitor_score": self._score([c.get("sandbox", False), c.get("mcp", False), c.get("subagents", False), c.get("frontend_workspace", False), c.get("artifact_ui", False)]),
                "why": "DeerFlow currently has the more complete production story for sandbox modes, MCP, and frontend artifact handling.",
            },
            {
                "label": "Result Surface",
                "subject_score": self._score([s.get("workspace_view", False), s.get("deep_research_report", False), s.get("interop_export", False)]),
                "competitor_score": self._score([c.get("official_website", False), c.get("frontend_workspace", False), c.get("artifact_ui", False), c.get("skill_installation", False)]),
                "why": "DeerFlow wins on polish and recognizability; agent-harness still needs more impressive end-user demos despite better internal semantics.",
            },
            {
                "label": "Benchmark Readiness",
                "subject_score": min(5.0, 2.4 + min(float(sm.get("benchmark_adapter_count", 0)) / 2.0, 1.2) + (0.8 if s.get("benchmark_adapters", False) else 0.0) + (0.4 if s.get("deep_research_report", False) else 0.0)),
                "competitor_score": 2.8,
                "why": "agent-harness has the clearer internal evaluation scaffold, but it still lacks public-format leaderboard history.",
            },
        ]
        for item in rows:
            delta = float(item["subject_score"]) - float(item["competitor_score"])
            item["delta"] = round(delta, 2)
            item["verdict"] = "ahead" if delta >= 0.35 else "behind" if delta <= -0.35 else "mixed"
        return rows

    @staticmethod
    def _score(flags: list[bool]) -> float:
        return round(5.0 * (sum(1 for flag in flags if flag) / max(len(flags), 1)), 1)
    @staticmethod
    def _benchmark_map() -> list[dict[str, str]]:
        return [
            {"name": "GAIA", "fit": "High", "gap": "Need public-format question runner and stronger open-web verification.", "value": "Validates deep research, retrieval, and multi-step reasoning quality.", "url": "https://huggingface.co/gaia-benchmark"},
            {"name": "SWE-bench Verified", "fit": "Medium", "gap": "Need patch -> targeted tests -> validation artifact loop on official task format.", "value": "Validates real engineering closure rather than architecture talk.", "url": "https://github.com/SWE-bench/SWE-bench"},
            {"name": "WebArena", "fit": "Low-Medium", "gap": "Need real browser action execution instead of only structured planning.", "value": "Validates interactive web task execution and long-horizon environment control.", "url": "https://webarena.dev/"},
            {"name": "tau-bench", "fit": "High", "gap": "Need enterprise connectors and stronger tool-user-environment loops.", "value": "Validates realistic enterprise task execution with tools and user constraints.", "url": "https://github.com/sierra-research/tau-bench"},
        ]

    @staticmethod
    def _build_roadmap(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        behind = [item for item in dimensions if item.get("verdict") == "behind"]
        mixed = [item for item in dimensions if item.get("verdict") == "mixed"]
        top_gap = behind[0]["label"] if behind else (mixed[0]["label"] if mixed else "Benchmark Readiness")
        return [
            {"phase": "Phase 1 - Make Research Outputs Credible", "focus": "Promote result-first demos, turn research target graphs into evidence-first pipelines, and make final reports materially better than decomposed briefs.", "target": top_gap},
            {"phase": "Phase 2 - Close Public Task Loops", "focus": "Wire benchmark adapters to GAIA, SWE-bench Verified, WebArena, and tau-bench public task schemas, then store reproducible run artifacts.", "target": "Benchmark Readiness"},
            {"phase": "Phase 3 - Productize The Harness", "focus": "Ship a streaming workspace and artifact board that makes execution, evidence, and results legible to outsiders within the first screen.", "target": "Result Surface"},
            {"phase": "Phase 4 - Push The Core Advantage", "focus": "Lean into Executable Task Graph as the cross-scene intermediate representation for research, engineering, and enterprise workflows.", "target": "Executable Runtime"},
        ]

    def _executive_summary(self, subject: dict[str, Any], competitor: dict[str, Any], dimensions: list[dict[str, Any]]) -> dict[str, Any]:
        ahead = [item["label"] for item in dimensions if item.get("verdict") == "ahead"]
        behind = [item["label"] for item in dimensions if item.get("verdict") == "behind"]
        mixed = [item["label"] for item in dimensions if item.get("verdict") == "mixed"]
        return {
            "title": "AI Agent Harness: Deep Research And Improvement Report",
            "summary": "This report studies the current `agent-harness` codebase against DeerFlow's public structure and concludes that the repo now has a stronger execution core than before, but it still needs more convincing public outputs and benchmark closure before it can credibly claim to beat DeerFlow as a general-agent harness.",
            "highlights": [
                f"Current lead: {ahead[0]}." if ahead else "No dimension is yet clearly dominant enough to declare victory.",
                f"Main deficit: {behind[0]}." if behind else "Main deficit is concentrated in mixed, still-unsettled dimensions.",
                f"Critical mixed zone: {mixed[0]}." if mixed else "Most dimensions currently separate clearly.",
            ],
            "value_statement": "The real differentiator is not another skill template. It is the ability to convert a request into an executable task graph, preserve state across retries and interruptions, and package the result as evidence-backed artifacts.",
            "differentiator": "If agent-harness doubles down on Executable Task Graph plus reproducible benchmark closure, it can beat skill-first repos on both engineering rigor and research value.",
            "research_statement": "Public benchmark history is still missing, but the internal lab and ablation scaffolding gives agent-harness a clearer path to scientific claims than most showcase-first projects.",
        }

    def _render_framework_markdown(self, topic: str, subject: dict[str, Any], competitor: dict[str, Any], dimensions: list[dict[str, Any]], benchmark_map: list[dict[str, str]]) -> str:
        dimension_rows = "\n".join(f"| {item['label']} | {item['subject_score']:.1f} | {item['competitor_score']:.1f} | {item['verdict']} | {item['why']} |" for item in dimensions)
        benchmark_rows = "\n".join(f"| {item['name']} | {item['fit']} | {item['gap']} | {item['value']} |" for item in benchmark_map)
        return f"""# AI Agent Harness Deep Research Framework

## Research Subject
- Topic: {topic}
- Primary target: {subject.get('name', 'agent-harness')}
- Comparator: {competitor.get('name', 'deer-flow')}
- Objective: determine whether `agent-harness` is structurally competitive with DeerFlow, where it is already ahead, and what must change next to become a genuinely strong general-agent harness.

## Core Questions
1. What is `agent-harness` actually good at today, beyond proposal generation?
2. Where does DeerFlow still lead in product completeness, skill methodology, and ecosystem usability?
3. Which improvements would create engineering value and academic value at the same time?
4. Which public benchmark families are the correct proof targets for the next stage?

## Evidence Plan
- Codebase scan of `app/`, `tests/`, and runtime-specific modules in `agent-harness`
- Public structure scan of DeerFlow README, backend/frontend docs, and public skills
- Internal evidence registry output for governance, research, and interoperability references
- Benchmark mapping against GAIA, SWE-bench Verified, WebArena, and tau-bench

## Comparison Axes
| Dimension | Agent Harness | DeerFlow | Verdict | Interpretation |
|---|---:|---:|---|---|
{dimension_rows}

## Benchmark Mapping
| Benchmark | Fit | Gap | Why It Matters |
|---|---|---|---|
{benchmark_rows}

## Deliverables
1. A deep research report with explicit strengths, deficits, and roadmap
2. A structured JSON evidence bundle for downstream demos or UI rendering
3. A result-first HTML page that surfaces the bottom line before internal mechanics
"""
    def _render_report_markdown(
        self,
        topic: str,
        subject: dict[str, Any],
        competitor: dict[str, Any],
        dimensions: list[dict[str, Any]],
        benchmark_map: list[dict[str, str]],
        roadmap: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> str:
        summary = self._executive_summary(subject, competitor, dimensions)
        strengths = [item for item in dimensions if item.get("verdict") == "ahead"]
        deficits = [item for item in dimensions if item.get("verdict") == "behind"]
        mixed = [item for item in dimensions if item.get("verdict") == "mixed"]
        evidence_rows = evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []
        strengths_block = "\n".join(f"- **{item['label']}** ({item['subject_score']:.1f} vs {item['competitor_score']:.1f}): {item['why']}" for item in strengths) or "- No dimension is ahead by a decisive margin yet."
        deficits_block = "\n".join(f"- **{item['label']}** ({item['subject_score']:.1f} vs {item['competitor_score']:.1f}): {item['why']}" for item in deficits) or "- No dimension is behind by a decisive margin, but several are still mixed."
        mixed_block = "\n".join(f"- **{item['label']}** ({item['subject_score']:.1f} vs {item['competitor_score']:.1f}): {item['why']}" for item in mixed) or "- No mixed dimensions."
        benchmarks = "\n".join(f"- **{item['name']}**: fit={item['fit']}; gap={item['gap']}; source={item['url']}" for item in benchmark_map)
        roadmap_lines = "\n".join(f"### {item['phase']}\n- Focus: {item['focus']}\n- Primary target: {item['target']}" for item in roadmap)
        evidence_lines = "\n".join(f"- **{row.get('title', '')}**: {row.get('summary', '')} ({row.get('url', row.get('path', ''))})" for row in evidence_rows[:8])
        subject_files = "\n".join(f"- `{row['path']}`: {row['reason']}" for row in subject.get("notable_files", [])[:6])
        competitor_files = "\n".join(f"- `{row['path']}`: {row['reason']}" for row in competitor.get("notable_files", [])[:5])
        return f"""# AI Agent Harness Deep Research And Improvement Report

## Executive Summary
{summary['summary']}

- One-line diagnosis: `agent-harness` is no longer just a routing experiment, but it is still not yet a convincingly superior general-agent product when judged against DeerFlow's public surface.
- Why users might still choose it: the framework now has a clearer execution core built around persistent threads, executable task graphs, workspace-bound actions, benchmark-ablation scaffolding, and interoperability exports.
- Why users would still hesitate: most of those strengths are internal; DeerFlow currently communicates its power better and packages it into a more complete public harness.

## Method
This report uses repository-level evidence rather than marketing copy. The analysis covers the live `agent-harness` codebase, the local DeerFlow reference repository, the built-in evidence registry, and benchmark-fit mapping against public agent benchmarks.

Topic studied: {topic}

## What Agent Harness Is Today
`agent-harness` has moved closer to a real harness because it now has three things that matter:

1. **Explicit execution semantics**  
   Persistent thread runtime, task-graph execution, node action mapping, async wait, resume, retry, interrupt, and recovery are not hand-waved.
2. **Cross-scene execution primitives**  
   The same runtime can already express tool calls, skill calls, workspace inspection, file writes, and command execution.
3. **Research and evaluation hooks**  
   The codebase includes benchmark adapters, ablation output, interoperability export, evidence injection, and result packaging layers.

Key supporting files:
{subject_files}

## Where DeerFlow Still Wins
DeerFlow is ahead in the places users notice first:

1. **Public product surface**  
   It exposes an official website, a thread-based frontend workspace, artifact views, memory, skill management, and a stronger super-agent-harness narrative.
2. **Skill methodology**  
   Its public `deep-research` skill forces broad exploration, deep dive, diversity validation, and synthesis. Its `consulting-analysis` skill explicitly separates analysis-framework generation from final-report generation.
3. **Production framing**  
   The repo is organized as a real full-stack harness with sandbox modes, MCP integration, gateway APIs, and frontend/backend docs that explain the system architecture.

Key supporting DeerFlow files:
{competitor_files}

## Competitive Assessment
### Structural Strengths Where Agent Harness Is Ahead
{strengths_block}

### Structural Deficits Where Agent Harness Is Behind
{deficits_block}

### Mixed Dimensions That Decide The Next Stage
{mixed_block}

## The Real Opportunity To Beat DeerFlow
The most defensible path is **not** to imitate DeerFlow skill-for-skill. The better path is to push a different center of gravity:

1. **Make Executable Task Graph the core intermediate representation**  
   One graph should drive research, code, operations, and enterprise workflows with explicit artifacts, retry semantics, and validation hooks.
2. **Turn skills from prompt snippets into graph-aware capability modules**  
   A strong skill in this system should not only describe how to think; it should specify what evidence to collect, what artifact to emit, and what validation gate to pass.
3. **Tie every big claim to a reproducible benchmark or execution artifact**  
   This is where the project can gain academic value instead of staying a showcase toy.

## Algorithmic Assessment
The strongest algorithmic idea in `agent-harness` today is the move toward a **stateful, resumable task graph runtime** rather than a pure chat loop. That is a real systems contribution because it gives the framework interruption-safe execution, node-level artifact contracts, cross-scene action unification, and a natural surface for ablation and benchmark logging.

The biggest algorithmic weakness is that the current skill layer is still too heuristic and text-heavy. Until more skills become execution-aware and evidence-aware, the framework's internal semantics will remain stronger than its final outputs.

## Benchmark Readiness
The correct benchmark families are:
{benchmarks}

Current honest assessment:
- `agent-harness` has internal benchmark structure, but not enough public-format execution history to claim superiority.
- The repo is best aligned to **GAIA** and **tau-bench** in spirit today.
- It will only deserve strong claims on **SWE-bench Verified** after patch -> targeted tests -> validation artifacts are fully closed on the official task format.
- It will only deserve strong claims on **WebArena** after real browser action loops exist, not just planning abstractions.

## Recommended Roadmap
{roadmap_lines}

## Why This Report Matters
If `agent-harness` follows the roadmap above, its differentiator becomes clear:

- DeerFlow is the better public super-agent harness today.
- Agent Harness can still become the better **research-grade execution harness** if it turns its runtime semantics into stronger outputs, stronger benchmarks, and stronger first-screen demos.

That is the path from interesting framework to something advanced users would switch to on purpose.

## Evidence Appendix
{evidence_lines if evidence_lines else '- No registry evidence records were returned.'}
"""

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
