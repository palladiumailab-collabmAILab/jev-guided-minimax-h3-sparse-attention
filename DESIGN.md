# Design

## Purpose

MiniMax H3のblock-sparse attention比率を、TypeSafe Jevの軽量な判断でdenoising stepごとに動的選択できるかを検証する実験prototype。

## Design principles

- **plannerとGPU hot pathを分離する。** Jevはkeep ratioを決定し、attention計算自体はローカルGPUで行う。
- **Cloudflareへ高次元tensorを送らない。** Q/K/Vは送信せず、小さいcontextと必要時のsampled statisticsだけを送る。
- **1 stepあたり1 inferenceを基本とする。** planner overheadがsparse化の利益を食わない構造にする。
- **typed bounded decisions。** keep ratioは許可集合に限定し、confidence不足時はconservative fallbackへ戻す。
- **mechanismと効果検証を分離する。** 実装できたことと、速度・品質改善が成立したことを混同しない。
- **公開endpointはfail-closed。** 認証がなければremote利用を拒否する。

## Non-goals

- 既存の速度向上率を再現済みとみなすこと。
- dense attentionと同等品質であると仮定すること。
- Cloudflare上でattention本体を計算すること。

## Validation

dense、fixed sparse、Jev-guidedを同一条件で比較し、wall-clock、VRAM、planner latency、selected ratios、qualityを記録する。
