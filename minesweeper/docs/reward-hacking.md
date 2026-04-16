# Reward Hacking & Reward Function Design for Minesweeper RL

## What is Reward Hacking

Reward hacking is when the agent finds unintended ways to maximize reward that don't align with the actual goal. A classic example: with few mines on the board, random clicking wins often enough that the agent learns "click anywhere" and gets heavily rewarded — but this strategy collapses when mine density increases.

## Curriculum Learning Pitfall (Negative Transfer)

A naive curriculum of increasing mine count (5 → 15 → 40) risks negative transfer:
- With 5 mines on 16x16, random clicking wins ~70%+ of games
- Agent deeply ingrains "click anywhere" as a strategy
- When switched to 40 mines, that strategy collapses but is hard to unlearn

**Solution:** Use fixed mine count from the start, or ensure the reward function penalizes random guessing when safe moves exist so the agent cannot exploit low mine density.

## Win Rate is a Misleading Metric

Win rate conflates luck and skill. On 16x16 with 40 mines, even a perfect logical player only wins ~60-70% due to forced guesses. Better behavioral metrics:

| Metric | What it measures |
|--------|-----------------|
| Survival length | Steps before dying — measures mine avoidance |
| Deductive move rate | % of clicks on provably safe cells |
| Unnecessary guess rate | Guessed when safe moves existed |
| Cells revealed per death | Quality of progress before dying |

## Reward Shaping Strategy

Split reward into named components and log each separately to tensorboard so you can see *why* performance is what it is:

```python
reward_reveal    += new_cells_opened          # base reward for progress
reward_deductive += 1.0  # if clicked provably safe cell
reward_penalty   -= 0.5  # if guessed when safe moves existed
reward_mine      -= 5.0  # extra if clicked a provably detectable mine
```

If `reward_deductive` never goes up, the agent is not learning deductive play regardless of win rate.

## Constraint-Based Safe/Mine Detection

First-order minesweeper constraints to classify moves:

1. Revealed cell shows `0` → all unrevealed neighbors are **provably safe**
2. Revealed cell shows `N` and has exactly `N` unrevealed neighbors → all are **provably mines**
3. Revealed cell shows `N` and already has `N` known mines as neighbors → remaining unrevealed neighbors are **provably safe**

Any click on a provably safe cell = deductive move (reward).
Any click on a random cell when safe moves exist = unnecessary guess (penalty).
Any click on a provably mine cell = ignoring obvious constraint (extra penalty).

## Testing Reward Functions

There is no formal unit testing equivalent, but these techniques help:

### Behavioral Probes
Hand-craft specific board states and verify the agent does the expected thing:
```python
# Set up board where cell (3,3) is provably safe
# Run model.predict() 100 times
# Assert agent clicks (3,3) > 80% of the time
```

### Policy Probing / Action Heatmap
Fix a board state, query `model.predict()` across many seeds, visualize the action distribution as a heatmap overlaid on the board. A good policy clusters on frontier cells, not spread randomly.

### Ablation Testing
Train two models — one with constraint bonus, one without — and compare behavioral metrics, not just win rate. If deductive move rate is higher with the bonus, the shaping is working even if win rate hasn't improved yet.

### Reward Decomposition Logging
Log each reward component separately in tensorboard. Lets you diagnose which behaviors are and aren't being learned.

## Other Techniques Beyond Reward Shaping

| Technique | Description | Effort |
|-----------|-------------|--------|
| Action masking (MaskablePPO) | Forbid clicking opened cells explicitly | Low |
| Curriculum learning | Start easy (few mines), increase difficulty | Medium |
| Auxiliary mine prediction | Secondary head predicts mine locations from `info["map"]` — forces CNN to build useful representations | Medium |
| Self-imitation learning | Replay best episodes more heavily | Medium |
| Hybrid rule-based + RL | Rule-based solver handles deductive moves, RL handles forced guesses | High |
| MCTS + RL (AlphaZero style) | Use tree search to generate higher quality training data | Very high |

## Realistic Win Rate Expectations

| Approach | Expected win rate (16x16, 40 mines) |
|----------|-------------------------------------|
| Random clicking | ~1-2% |
| PPO MlpPolicy, 400k steps | ~5-10% |
| PPO CnnPolicy, 2M steps | ~15-25% |
| PPO CnnPolicy + reward shaping + curriculum | ~30-40% |
| Perfect logical solver (no guessing) | ~60-70% (rest are forced guesses) |
| Theoretical maximum | ~60-70% |

Getting above ~40% likely requires a hybrid approach where a constraint solver handles all deductive moves and RL only makes decisions on genuinely ambiguous positions.
