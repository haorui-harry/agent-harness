# Visual Payload Contract

This file describes how to consume `harness-visual` and `harness-showcase` output in a front-end.

## Single Run Payload (`harness-visual`)

Top-level fields:

- `kpis`: numeric cards (`value_index`, `reliability`, `safety`, `innovation`, etc.)
- `radar`: `{ labels: string[], values: number[] }`
- `timeline`: step bars with `start_ms`, `end_ms`, `status`
- `discovery_board`: candidate tools with score/risk/novelty
- `security_board`: preflight decision + step actions
- `live_agent_board`: real-model status, calls, analysis, critique
- `tool_network`: `{ nodes, links }` for force graph
- `hero_cards`: three short cards for first-screen storytelling
- `first_screen_blueprint`: opinionated layout metadata
- `event_stream`: replay events (`run_started`, `security_preflight`, `step_started`, etc.)

## Multi Scenario Payload (`harness-showcase`)

Top-level fields:

- `overview`: aggregate pack metrics
- `comparison`: compact table + best scenario per metric
- `scenarios`: full run summaries and value cards
- `visual_payloads`: array of single-run payloads
- `hero_story`: one-line narrative points

## Suggested Front-End Flow

1. Render `first_screen_blueprint.hero.kpi_cards`.
2. Render radar using `radar.labels` + `radar.values`.
3. Render timeline from `timeline`.
4. Render discovery as scatter bubbles (`score` x `novelty`).
5. Render security lane with `preflight_action` and per-step actions.
6. Render force graph using `tool_network.nodes` and `tool_network.links`.
