<p align="right">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/Language-English-0f766e?style=for-the-badge"></a>
  <a href="./README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:081826,35:0F766E,70:2563EB,100:F59E0B&height=220&section=header&text=Agent%20Harness&fontSize=54&fontColor=ffffff&desc=%E6%8A%8A%E4%B8%80%E4%B8%AA%E8%AF%B7%E6%B1%82%E5%8F%98%E6%88%90%E5%8F%AF%E5%AE%A1%E8%AE%A1%E7%9A%84%20agent%20%E4%BA%A7%E5%93%81.&descAlignY=68&animation=fadeIn" alt="Agent Harness banner"/>
</p>

<p align="center">
  <b>Agent Harness 是一个面向路由、执行、评测、展示和互操作的智能体产品引擎。</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Routing-robust_frontier-0f766e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Execution-Harness%20Engine-2563eb?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Eval-harness--lab-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Interop-OpenAI%20%7C%20Anthropic-111827?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-Orchestrated-0B1020"/>
  <img src="https://img.shields.io/badge/CLI-Typer-111827"/>
  <img src="https://img.shields.io/badge/Tests-81%20passed-16a34a"/>
</p>

---

## 框架图

![Agent Harness Framework Diagram](docs/d5_260402_5995__QakBh2Z.jpg)

这个项目的真正主线是：

`用户请求 -> Agent Router -> Agent Council -> Skill Router -> Harness Engine -> 证据层 / 实验层 / 展示层 / 互操作层`

它不是普通的路由器，也不是简单的 skill 仓库，而是一个把请求转化为产品级交付物的系统。

---

## 一句话概括

Agent Harness 是一个 agent operating system：它会在不确定性下选择更稳健的 skill 组合，用 harness 执行层完成治理、记忆和工具执行，用 research lab 做 benchmark 与发布门禁，最后把结果打包成可展示、可发布、可互操作的产品级 bundle。

---

## 为什么做这个项目

现在很多框架往往只在一个方向强：

- 偏 flow 的系统擅长 orchestration，但不擅长证明自己为什么这么路由
- 偏 research 的系统深度很强，但不擅长产品化交付
- 偏 skill hub 的系统生态很广，但治理、评测、发布门禁薄弱

Agent Harness 的目标就是把这三者统一起来：

1. 更好地路由
2. 带证据地执行
3. 在发布前完成 benchmark
4. 输出可展示、可发布的产物
5. 把能力导出到外部生态

---

## Demo 展示区

### Demo Gallery

已发布的 demo 快照放在 `docs/demo/` 下。运行时生成的 `reports/` 结果保留为本地产物，并且被刻意加入了 gitignore。

查看完整快照入口：[docs/demo/README.zh-CN.md](./docs/demo/README.zh-CN.md)

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Live 方案展示页</h3>
      <p><img src="./docs/demo/live/showcase.png" alt="Live showcase screenshot"></p>
      <p>发布会风格 HTML，优先展示方案摘要、rollout phases、evidence、framework comparison 和 agent comparison。</p>
      <p><a href="./docs/demo/live/showcase.html">打开 HTML 快照</a></p>
      <p><a href="./docs/demo/live/press-brief.md">打开 Press Brief</a></p>
    </td>
    <td width="33%" valign="top">
      <h3>Baseline Launch Bundle</h3>
      <p><img src="./docs/demo/press/showcase.png" alt="Baseline showcase screenshot"></p>
      <p>可复现的 baseline 展示产物，适合稳定 demo 和 CI 场景。</p>
      <p><a href="./docs/demo/press/showcase.html">打开 HTML 快照</a></p>
      <p><a href="./docs/demo/press/showcase.json">打开 JSON</a></p>
    </td>
    <td width="33%" valign="top">
      <h3>生态互操作导出</h3>
      <p><img src="./docs/demo/live/showcase.png" alt="Interop-backed launch artifact"></p>
      <p>同一套能力可导出到 OpenAI / Anthropic skill 生态。</p>
      <p><a href="./docs/demo/live/interop_bundle.json">打开 Interop Bundle</a></p>
    </td>
  </tr>
</table>

### 当前 Demo 在展示什么

当前 showcase 页面优先展示：

1. 方案摘要
2. phased rollout
3. evidence bundle
4. framework comparison
5. agent comparison
6. 内部 skill / tool 附录

最近一次 agent 对比结果：

- Winner: `PlannerAgent`
- Runner-up: `ResearchAgent`
- Score gap: `0.0019`

这说明这个框架不会把接近的决策隐藏掉，而会把它显式展示出来。

---

## Feature 对比区

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>普通仓库通常停留在哪</h3>
      <ul>
        <li>单 agent 选择</li>
        <li>top-k 或 relevance-only skill 选择</li>
        <li>只有答案，没有正式证据</li>
        <li>评测零散</li>
        <li>生态封闭</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>Agent Harness 增加了什么</h3>
      <ul>
        <li>agent council 和协作感知路由</li>
        <li><code>robust_frontier</code> 不确定性下的 skill 选择</li>
        <li>trace、response contract、routing analysis</li>
        <li><code>harness-lab</code> leaderboard 与 release gate</li>
        <li>studio showcase 与 OpenAI / Anthropic interop</li>
      </ul>
    </td>
  </tr>
</table>

### 能力矩阵

| 能力 | 常见仓库 | Agent Harness |
|---|---|---|
| Agent routing | single winner | primary agent + agent council |
| Skill routing | relevance/top-k | `robust_frontier` over reliability, uncertainty, downside |
| Execution | tool loop | harness engine with tools, memory, guardrails, live API |
| Evidence | partial logs | trace + response contract + routing analysis |
| Evaluation | ad hoc | `harness-lab` with leaderboard + release gate |
| Product output | answer only | HTML + JSON + press brief + manifest |
| Ecosystem | closed | OpenAI / Anthropic skill interop |

---

## 架构说明

### Routing Layer

- `Agent Router`：根据 intent、complexity、risk 对候选 agent 打分
- `Agent Council`：在需要协同时扩展能力池
- `Skill Router`：使用 `robust_frontier` 选择最终 skill 组合

### Skill Layer

`skill` 是框架中的最小能力单元。它不只是一个 prompt，还带有：

- metadata
- strengths / weaknesses
- cost
- reliability
- uncertainty signals
- category / tier / synergies / conflicts

skill 可以来自：

- 本地内置 registry
- marketplace / ecosystem 来源
- 运行时动态 external skill
- OpenAI / Anthropic 兼容导出 bundle

### Harness Layer

`harness` 是围绕 routing 的执行 operating system，负责：

- tool discovery
- recipe selection
- memory reuse
- guardrails
- live API enhancement
- evaluation
- value scoring
- visual payload generation
- report generation
- research lab benchmarking

这也是为什么这个项目不是普通 router。

### Studio Layer

`studio-showcase` 会把运行时证据转成发布会风格的产品页：

- 方案页
- benchmark 对比
- agent comparison
- generated result excerpt
- 内部 skill / tool appendix

### Interop Layer

框架可以把能力导出到外部生态，因此它不是封闭系统，而是一个可被消费的能力平台。

---

## 方法亮点

### Risk-Calibrated Frontier Routing

skill selector 不是只按 relevance 排序，而是联合考虑：

- relevance
- diversity
- redundancy penalty
- synergy
- empirical reliability
- uncertainty penalty
- downside risk

核心指标包括：

- `robust_expected_utility`
- `robust_worst_case_utility`
- `avg_uncertainty`

### Research-Grade Release Gating

`harness-lab` 会运行可重复 scenario suite，并输出：

- leaderboard
- pass rate
- value index
- security alignment
- 发布结论：`go / caution / block`

---

## Quick Start

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 跑核心路由

```bash
python -m app.main run "Compare two rollout plans and highlight governance risk" --mode deep --contract
```

### 3. 跑 harness 执行层

```bash
python -m app.main harness "Prepare a governance-ready execution memo" --mode balanced
```

### 4. 生成 showcase

```bash
python -m app.main studio-showcase "Design a flagship AI operating plan" --mode deep --lab-preset broad --tag flagship
```

### 5. 生成 launch demo

```bash
python -m app.main launch-demo --output-dir reports/launch_demo --tag press
```

### 6. 生成真实 API demo

先设置环境变量：

```bash
set AGENT_HARNESS_MODEL_BASE_URL=https://your-endpoint/v1
set AGENT_HARNESS_MODEL_API_KEY=your_api_key
set AGENT_HARNESS_MODEL_NAME=your_model
```

再执行：

```bash
python -m app.main launch-demo --output-dir reports/live_launch_demo --tag live --live-agent --max-model-calls 6
```

### 7. 运行 research lab

```bash
python -m app.main harness-lab --preset broad --repeats 2 --output reports/lab.json
python -m app.main harness-lab-product --preset broad --repeats 2 --tag release --output-dir reports
```

### 8. 导出 interop bundle

```bash
python -m app.main skills-interop-export --framework all --output-dir reports/skills_interop
```

### 9. 跑测试

```bash
pytest -q
```

---

## 命令地图

| 目标 | 命令 |
|---|---|
| 基础 routing | `python -m app.main run "<query>"` |
| trace 与 reasoning | `python -m app.main trace "<query>" --views` |
| harness 执行 | `python -m app.main harness "<query>"` |
| value card | `python -m app.main harness-value "<query>"` |
| visual payload | `python -m app.main harness-visual "<query>" --output reports/visual.json` |
| showcase | `python -m app.main studio-showcase "<query>" --tag demo` |
| launch demo | `python -m app.main launch-demo --tag press` |
| research lab | `python -m app.main harness-lab --preset broad` |
| interop export | `python -m app.main skills-interop-export --framework all` |

---

## 仓库地图

### Core runtime

- `app/routing/` - agent routing, skill routing, complementarity, robust frontier
- `app/policy/` - system modes, governance, robustness policy
- `app/personality/` - profiles and adaptation
- `app/coordination/` - conflict, dissent, consensus
- `app/core/` - state, contracts, shared protocol

### Execution and productization

- `app/harness/` - execution engine, manifests, tools, reports, lab, value, visuals
- `app/studio/` - flagship showcase builder and launch packaging
- `app/tracing/` - trace rendering and quality analysis
- `app/utils/` - console and display helpers

### Skills and ecosystem

- `app/skills/` - built-in and external skill registry
- `app/ecosystem/` - marketplace signals, providers, imports
- `app/skills/interop.py` - OpenAI / Anthropic compatibility export

### Assets and artifacts

- `docs/` - architecture prompts and images
- `reports/` - generated demos, lab bundles, showcases
- `tests/` - regression coverage

---

## 当前状态

- `robust_frontier` routing 已实现
- agent council routing 已实现
- harness engine 和 research lab 已实现
- studio showcase 和 launch demo 已实现
- OpenAI / Anthropic interop export 已实现
- 当前本地测试结果：`81 passed`

---

## 推荐第一次体验方式

```bash
python -m app.main launch-demo --output-dir reports/launch_demo --tag press
python -m app.main studio-showcase "Design a flagship AI operating plan" --mode deep --lab-preset broad --tag studio
```

然后打开：

- `docs/demo/press/showcase.html`
- `docs/demo/live/showcase.html`
- `docs/demo/live/press-brief.md`

这些文件比一段 CLI trace 更能体现项目的产品形态。
