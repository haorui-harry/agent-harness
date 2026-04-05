<p align="right">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/Language-English-0f766e?style=for-the-badge"></a>
  <a href="./README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:081826,35:0F766E,70:2563EB,100:F59E0B&height=220&section=header&text=Agent%20Harness&fontSize=54&fontColor=ffffff&desc=%E4%B8%80%E4%B8%AA%E9%9D%A2%E5%90%91%E9%80%9A%E7%94%A8%E4%BB%BB%E5%8A%A1%E7%9A%84%20agent%20runtime%EF%BC%8C%E8%83%BD%E7%95%99%E4%B8%8B%E7%9C%9F%E6%AD%A3%E5%8F%AF%E4%BA%A4%E4%BB%98%E7%9A%84%E4%BA%A7%E7%89%A9%E3%80%82&descAlignY=68&animation=fadeIn" alt="Agent Harness banner"/>
</p>

<p align="center">
  <b>Agent Harness 是一个 thread-first 的通用智能体运行时：它按能力规划任务，在需要时调用 skill、tool 和 workspace action，并把结果落成真正可打开、可审查、可继续执行的交付物。</b>
</p>

---

## 一句话

Agent Harness 会把一个开放任务变成“主交付物 + 证据/工件轨道”，而不是只吐出一段答案字符串。

## 为什么做这个项目

最优秀的通用 agent 往往不需要非常复杂。

真正重要的是少数几个强原语：

- 一个持久 thread
- 一个按 capability 规划的 planner，而不是写死流程
- 一个真实 workspace，可以落文件、跑动作、留痕迹
- 一个用户真正会打开的主交付物
- 一条支持审计、恢复、追踪和继续工作的 artifact / evidence 轨道

Agent Harness 就围绕这几个东西收敛。

## 它最终能产出什么

根据任务不同，主交付物可以是：

- 研究报告
- patch draft
- 工程 handoff memo
- benchmark manifest / run config
- rollout plan / launch memo
- slide deck plan / webpage blueprint
- delivery bundle

配套工件可以包括：

- evidence bundle
- workspace findings
- validation plan / execution trace
- source matrix
- OpenAI / Anthropic 风格互操作导出

## 核心模型

`Request -> Thread -> Capability Planner -> Skills / Tools / Workspace Actions -> Main Deliverable -> Evidence + Artifact Bundle`

这个结构比“给每类任务写一条固定工作流”更通用，也更容易继续变强。

框架不应该先假设“这是 research flow”或“这是 code flow”，而应该先看：

- 任务要什么主交付物
- 当前有哪些能力可用
- 还缺哪些证据或工件
- workspace 里有什么真实上下文

## Agent 主循环

整个 runtime 围绕一个很短的循环：

1. 打开或恢复一个 thread
2. 从任务里推断主交付物以及还缺哪些 channel
3. 只在任务真正需要时再看 skill、tool、web 信息或 workspace
4. 在 thread workspace 里执行一个小型 task graph
5. 产出一个主交付物，再附上可审查的证据和后续工件

这部分最值得保护。只要这条循环变得过于复杂，系统就会从“通用 agent”重新退化成“一堆工作流拼装”。

## 这个项目和常见框架的差别

### 1. 主交付物优先

最重要的结果是用户要的交付物本身。

不是分数。
不是 bundle。
不是 planner trace。

后者都保留，但它们应该服务主结果，而不是抢主结果的戏份。

### 2. 默认按通用任务设计

它的目标不是“研究工作流引擎”或“代码工作流引擎”。

它从 task spec、capability、证据缺口和 workspace 状态出发，因此同一套 runtime 能覆盖代码、研究、运营和混合任务。

### 3. Thread + Workspace + Recovery

每个任务都能进入持久 thread，天然支持：

- resume
- retry
- interrupt
- recover
- workspace artifact
- event stream / snapshot export

### 4. Skill 有用，但不绑死产品

skill 是能力模块，不是整个产品本身。运行时还可以把能力导出成互操作目录，方便外部生态消费。

### 5. 按主交付物闭环

现在 runtime 在收尾时会先看“主交付物到底是什么”，再决定用哪个 synthesis surface 去完成它。

这和把所有任务都按一个“任务类别 ending”去收口不一样。patch draft、research brief、slide deck plan、ops runbook，本来就不该用同一种方式闭环。

---

## Demo 展示

仓库中可直接打开的 demo 在 `docs/demo/`。

完整索引见：[docs/demo/README.zh-CN.md](./docs/demo/README.zh-CN.md)

### 实时金融发布包

- 主交付物：[docs/demo/live/deliverable.md](./docs/demo/live/deliverable.md)
- 展示页：[docs/demo/live/showcase.html](./docs/demo/live/showcase.html)
- Press Brief：[docs/demo/live/press-brief.md](./docs/demo/live/press-brief.md)

### 企业 rollout 套件

- 主交付物：[docs/demo/enterprise/deliverable.md](./docs/demo/enterprise/deliverable.md)
- 展示页：[docs/demo/enterprise/showcase.html](./docs/demo/enterprise/showcase.html)
- Press Brief：[docs/demo/enterprise/press-brief.md](./docs/demo/enterprise/press-brief.md)

### 研究晋升包

- 主交付物：[docs/demo/research/deliverable.md](./docs/demo/research/deliverable.md)
- 展示页：[docs/demo/research/showcase.html](./docs/demo/research/showcase.html)
- Press Brief：[docs/demo/research/press-brief.md](./docs/demo/research/press-brief.md)

---

## Quick Start

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 创建一个持久 thread

```bash
python -m app.main agent-thread-create "General Agent Demo"
```

### 3. 跑一个通用任务

```bash
python -m app.main agent-thread-run <thread_id> "请写一份关于 agent runtime 可靠性的深度研究报告" --target auto
```

### 4. 查看 thread workspace

```bash
python -m app.main agent-thread-workspace-view <thread_id> --output reports/thread_view
```

### 5. 生成 showcase

```bash
python -m app.main studio-showcase "为一个受监管 AI copilot 设计发布计划" --tag demo
```

### 6. 接入 live model

请把密钥放在环境变量或命令行参数里，不要写进仓库。

```bash
set AGENT_HARNESS_MODEL_BASE_URL=https://your-endpoint/v1
set AGENT_HARNESS_MODEL_API_KEY=your_api_key
set AGENT_HARNESS_MODEL_NAME=your_model
python -m app.main harness-live "写一份有 benchmark 支撑的研究简报"
```

### 7. 运行测试

```bash
pytest -q
```

---

## 常用命令

| 目标 | 命令 |
|---|---|
| 创建 thread | `python -m app.main agent-thread-create "My Task"` |
| 列出 threads | `python -m app.main agent-threads` |
| 运行通用 super-agent | `python -m app.main agent-thread-run <thread_id> "<query>" --target auto` |
| 直接执行 task graph | `python -m app.main agent-thread-exec-task <thread_id> "<query>" --target general` |
| 导出 workspace 视图 | `python -m app.main agent-thread-workspace-view <thread_id>` |
| 导出 thread snapshot | `python -m app.main agent-thread-export <thread_id>` |
| live model 模式运行 | `python -m app.main harness-live "<query>"` |
| 生成 code mission pack | `python -m app.main harness-code-pack "<query>" --workspace .` |
| 跑 benchmark suite | `python -m app.main benchmark-suite --adapters routing_internal,lab_daily` |
| 生成 showcase | `python -m app.main studio-showcase "<query>" --tag demo` |
| 导出 skills interop | `python -m app.main skills-interop-export` |

---

## 仓库结构

### Runtime

- `app/harness/`：planner、live orchestration、evaluation、report、visuals
- `app/agents/`：thread runtime、workspace action mapper、scheduler、sandbox、workspace view
- `app/skills/`：内置 skill、skill package、interop export
- `app/core/`：task spec、capability graph、contract、共享状态

### Product Surface

- `app/studio/`：showcase 和发布包装层
- `docs/demo/`：随仓库提交的 demo 快照
- `reports/`：本地运行时生成的输出

### Validation

- `tests/`：runtime、showcase、planner、live orchestration 的测试

---

## 当前状态

Agent Harness 现在最强的地方是：

- persistent thread
- artifact-producing execution
- evidence-aware synthesis
- inspectable output

接下来最重要的不是继续加层，而是：

- 让更多产品形态复用同一套 generic runtime，而不是继续加包装层
- 继续削减 query 分类捷径，更多依赖 task spec 和 artifact contract
- 强化长时 thread 执行，但不把核心循环做重

它还没有完成，但方向已经更接近真正的通用 agent runtime，而不是一个 showcase workflow。
