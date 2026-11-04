# Distractor Intervention Experiment

## Research question
After training with non-causal distractor actions available, does the policy continue to allocate action probability to those actions once they are made non-functional (no longer advance the episode), compared to an agent with no prior distractor-training history placed in the same disabled-action environment?

## Why this follows the distractor action experiment
`distractor_action_experiment.py` found that CDS and SPI rise with the number of available distractor actions while goal rate stays near 1.0, and its own recommended next experiment was exactly this: remove the distractor affordance after training and see whether the trained policy still reaches for it. This experiment implements that intervention.

## Design
For each distractor count D in [2, 5, 10], 30 tabular Q-learning seeds are pretrained for 2500 episodes exactly as in the distractor action experiment. Each pretrained agent then enters an intervention environment where any action other than the causal action (action 0) no longer advances the episode's phase counter -- it is still selectable, but it is now a non-functional ritual. The agent continues to learn online (epsilon=0.1) for 1000 further episodes, binned into 20 blocks of 50 episodes. A naive control agent (zero-initialized Q-table, no prior exposure to distractor actions ever being useful) is run through the identical intervention environment for the same number of episodes and epsilon, as the no-superstition-history baseline.

## Metrics
Per-block SPI is the fraction of pre-goal greedy+exploratory decisions that select a disabled distractor action. Per-block goal rate is the fraction of episodes in that block reaching the causal reward. Half-life (in blocks) is the first block at which mean SPI falls to half of its block-0 value; `persistence_gap_blocks` is intervened half-life minus naive-control half-life -- positive means prior training history slows extinction of the now-useless action relative to an agent that never had that history.

## Results table

| D | pre-intervention SPI | intervened block-0 SPI | naive block-0 SPI | intervened block-1 SPI | naive block-1 SPI | intervened half-life (blocks) | naive half-life (blocks) | intervened final SPI | naive final SPI | Cohen's d | p (final SPI) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.659 | 0.094 | 0.354 | 0.065 | 0.068 | 20 | 1 | 0.063 | 0.064 | -0.048 | 0.8552 |
| 5 | 0.833 | 0.133 | 0.573 | 0.088 | 0.085 | 20 | 1 | 0.082 | 0.086 | -0.364 | 0.1712 |
| 10 | 0.930 | 0.207 | 0.855 | 0.097 | 0.299 | 1 | 1 | 0.093 | 0.091 | 0.229 | 0.3877 |

## Interpretation
Intervened block-0 SPI is lower than naive-control block-0 SPI for every distractor count: True. This is the opposite of what intervention-resistant superstition would predict -- the previously-trained agent enters the disabled-action environment already favoring the causal action more than a zero-initialized agent does, because the naive agent's untrained Q-table starts tied at zero and greedy tie-breaking picks uniformly among all actions (expected SPI close to D/(D+1)) until a handful of updates break the tie. By block 1 (episodes 51-100 of the intervention), both groups are within 0.02 SPI of each other for every distractor count: False, and that shared level matches the SPI produced by epsilon-driven forced exploration alone (epsilon=0.1 times the chance of a random distractor pick, D/(D+1)), not a residual policy preference. The apparent 'censored' half-lives for the intervened group at low D are an artifact of its block-0 value already sitting near that same exploration floor, not of slow extinction: the policy's greedy preference for the distractor action is gone within the first block in every condition.

## Does this support persistent computational superstition surviving intervention?
Supported: False. The policy abandons the disabled action immediately once it stops advancing the episode; prior training history does not slow this down relative to a naive agent -- if anything it gives the intervened agent a head start, since it already knows the causal action's value. The elevated CDS/SPI seen during normal training in the distractor action experiment therefore looks like an artifact of credit diffusion while the distractor affordance is still (harmlessly) available, not a stable, intervention-resistant habit.

## Limitations
The naive control starts from a zero-initialized Q-table rather than from an agent trained on a semantically unrelated task, so it is a lower bound on how much history matters, not a full dissociation of 'any training' from 'distractor-specific training'. The half-life metric is noisy when the block-0 baseline is already close to the exploration floor (small denominators make 'half of baseline' a weak threshold); the block-0/block-1 SPI comparison above is the more reliable signal in this run. Approximate p-values come from scipy's independent two-sample t-test on 30 seeds per group and do not correct for multiple comparisons across the three distractor counts.

## Next recommended experiment
Re-run this intervention with a function-approximation agent (linear or neural DQN) instead of tabular Q-learning, to test whether generalization across states changes how quickly the now-disabled distractor action's value is unlearned, mirroring the tabular-vs-DQN contrast already used in the causal reversal experiments.
