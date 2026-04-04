**Executive Summary**  
Agent-harness aims to bridge academic research and production deployment for agent-based AI systems, but currently lacks rigorous experimental design, reproducible benchmarking, and evidence-based validation standards. Key gaps include fragmented benchmark integration (tau-bench, SWE-bench), inconsistent evidence packet implementation, and missing statistical significance frameworks. High-leverage improvements—standardized benchmark manifests, automated run configuration, MCP-based tool interoperability, and promotion criteria enforcement—can address these gaps. This report outlines a concrete 12-week roadmap to achieve 95% reproducibility, sub-4-hour experiment setup, and full evidence compliance, positioning agent-harness as a leading applied research platform.

---

**Current System Assessment**  
Agent-harness operates with fragmented evaluation workflows. Internal promotion criteria require joint satisfaction of reproducibility, benchmark stability, and operating constraints, yet these are not systematically enforced. Evidence packet templates exist but are inconsistently applied, forcing manual assembly of metrics, controls, and citations for launch-readiness. Benchmark integration is partial: tau-bench (enterprise tool-using agents) and SWE-bench (verifiable code changes) provide realistic task evaluation but lack automated cross-validation in agent-harness’s testing loops. The absence of a unified evidence dossier—linking quantitative deltas, control comparisons, and citation-backed rationales—slows research-to-production transitions and increases rollout latency.

---

**Comparative Analysis**  
- **tau-bench**: Focuses on enterprise tool-using agents and long-horizon workflows. Agent-harness would benefit from its realistic task design but lacks automated cross-validation and durability checks.  
- **SWE-bench**: Measures code-change resolution against real GitHub issues. Its verifiable evaluation rigor is not mirrored in agent-harness’s validation suite, missing checks for regression and edge-case robustness.  
- **LangGraph**: Provides stateful agent orchestration and durability patterns. Agent-harness could adopt its loop execution models to improve experimental consistency but currently lacks equivalent state management.  
- **Model Context Protocol (MCP)**: Defines standards for agent-tool interoperability. Agent-harness’s tool integration is less composable, limiting external capability composition.  
- **OpenAI Evals/Anthropic Claude Console**: Offer structured evaluation frameworks with statistical significance thresholds. Agent-harness lacks equivalent p-value requirements (e.g., p < 0.05) for performance claims and effect size validation.

---

**Failure Modes**  
1. **Non-reproducible Experiments**: Missing standardized benchmark manifests and versioned run configurations prevent consistent replication across teams.  
2. **Unvalidated Performance Claims**: No statistical significance framework risks promoting agents based on noisy or non-significant metrics, without power analysis or confidence intervals.  
3. **Tool Interoperability Gaps**: Limited MCP integration restricts agent capability composition and external tool usage, hindering enterprise readiness.  
4. **Evidence Fragmentation**: Decision-makers manually collate metrics, controls, and citations from separate systems, increasing rollout latency and error risk.  
5. **Benchmark Isolation**: tau-bench and SWE-bench are not embedded in continuous testing, delaying enterprise-readiness signals and missing robustness checks.  
6. **Resource and Cost Blind Spots**: No computational budgeting or cost tracking for large-scale benchmarking, risking unsustainable experiment scaling.

---

**High-Leverage Improvements**  
- **Standardized Benchmark Manifest**: Define a machine-readable JSON schema for experiment tracking, ensuring reproducibility and version control across runs.  
- **Automated Run Configuration Generator**: Templated multi-scenario test suites to reduce setup time from days to under 4 hours.  
- **Statistical Significance Framework**: Implement A/B testing with p < 0.05 thresholds, power analysis, and confidence intervals for all performance claims.  
- **Evidence Packet Automation**: Auto-generate release packets combining metrics, controls, citations, risks, and rollback conditions in PDF/JSON formats.  
- **MCP Integration**: Adopt Model Context Protocol for seamless agent-tool interoperability, supporting 10+ external tools within 12 weeks.  
- **Promotion Criteria Enforcement**: Systematically gate candidate promotion on reproducibility (3-run checks), stability (confidence interval thresholds), and operating constraints (latency, cost).  
- **Resource and Cost Dashboard**: Implement computational budgeting and real-time cost tracking for benchmark runs.

---

**Benchmark Plan**  
- **Reproducibility**: Achieve 95% benchmark reproducibility across 3 consecutive runs via versioned manifests and run configurations.  
- **Velocity**: Reduce experiment setup time to under 4 hours through standardized configurations and CLI tooling.  
- **Statistical Rigor**: Establish p < 0.05 significance thresholds with power analysis (β ≥ 0.8) for all performance deltas.  
- **Cross-Validation**: Integrate tau-bench and SWE-bench suites to validate enterprise readiness, including regression and edge-case testing.  
- **Evidence Compliance**: Maintain 100% evidence packet completion for promoted experiments, auto-linking metrics, controls, and citations.  
- **Tool Interoperability**: Support 10+ external tools via MCP within 12 weeks, with security and privacy checks for integrated tools.  
- **Cost Efficiency**: Limit computational cost overruns to <10% of budget through real-time resource tracking.

---

**12-Week Roadmap**  
**Weeks 1–4: Benchmark Infrastructure**  
- Deliverable 1: Standardized benchmark manifest specification (JSON Schema) with version control.  
- Deliverable 2: Automated run configuration generator (CLI tool) for multi-scenario testing.  
- Deliverable 3: tau-bench and SWE-bench integration hooks with regression testing.  
- Deliverable 4: Baseline performance measurements across 5 agent tasks, including cost tracking.

**Weeks 5–8: Experimental Rigor**  
- Deliverable 5: Statistical significance framework with confidence intervals and power analysis.  
- Deliverable 6: A/B testing infrastructure for multi-agent scenarios, including placebo controls.  
- Deliverable 7: Evidence packet automation (PDF/JSON output) with auto-citation linking.  
- Deliverable 8: Reproducibility verification protocol (3-run checks) and adversarial testing suite.

**Weeks 9–12: Production Bridge**  
- Deliverable 9: MCP integration for tool interoperability, with security and privacy validation.  
- Deliverable 10: Promotion criteria enforcement system (CI/CD gates) for reproducibility, stability, and constraints.  
- Deliverable 11: Risk assessment dashboard with rollback conditions and real-time monitoring.  
- Deliverable 12: 12-month research backlog prioritization framework, incorporating cost-benefit analysis.

---

**Open Questions**  
- How to handle benchmark drift in long-horizon workflows without manual recalibration?  
- What evidence standards suffice for regulatory compliance (e.g., GDPR, HIPAA) in enterprise deployments?  
- Can agent-harness adopt LangGraph’s durability patterns without sacrificing experimental flexibility?  
- How to scale reproducibility checks across distributed research teams with varying infrastructure?  
- What metrics best capture “production readiness” beyond task performance (e.g., latency, throughput, user satisfaction)?  
- How to incorporate human evaluation and qualitative assessment alongside automated metrics?

---

**Sources**  
- Experiment Promotion Criteria (internal://research/experiment-promotion-criteria)  
- tau-bench (https://github.com/sierra-research/tau-bench)  
- Model Context Protocol Architecture (https://modelcontextprotocol.io/specification/2025-06-18/architecture/index)  
- Evidence Packet Template (internal://cross/evidence-packet-template)  
- LangGraph Overview (https://docs.langchain.com/oss/python/langgraph/overview)  
- SWE-bench (https://github.com/SWE-bench/SWE-bench)

## Evidence References

- internal://research/experiment-promotion-criteria
- https://github.com/sierra-research/tau-bench
- https://modelcontextprotocol.io/specification/2025-06-18/architecture/index
- internal://cross/evidence-packet-template
- https://docs.langchain.com/oss/python/langgraph/overview
- https://github.com/SWE-bench/SWE-bench

## Openable Files

- Primary Deliverable: studio_deliverable_research.md
- Showcase HTML: studio_showcase_research.html
- Showcase JSON: studio_showcase_research.json
- Press Brief: studio_press_brief_research.md
- Bundle Manifest: studio_bundle_manifest_research.json
- Interop Bundle: studio_interop_research/index.json
