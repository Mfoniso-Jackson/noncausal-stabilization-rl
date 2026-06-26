# Linear Feature Ablation Experiment

## Research question
Was the previous linear DQN proxy-persistence result genuine proxy reliance, or an artifact of the linear feature representation?

## Design
The hidden-state causal reversal environment, phase schedule, reward rule, action space, DQN hyperparameters, seeds, and evaluation protocol are unchanged. Only the observation feature map is varied.

## Feature conditions
- phase_only: phase and target/pre-target flags, no proxy cue.
- proxy_only: proxy cue only.
- phase_plus_proxy: phase and proxy cue.
- phase_sin_cos_plus_proxy: phase, sinusoidal phase encoding, and proxy cue.
- full_current_features: previous full linear-DQN feature set.
- full_features_with_noise: full feature set plus two irrelevant noise features.

## Validity checks
hidden_reward_state is never observed. Action 1 never directly causes reward. Reward only depends on hidden_reward_state and action 0 at the target phase. Action-space size remains fixed at 4. Random baseline is included.

## Results table

| feature_condition | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |
|---|---|---:|---:|---:|---:|---:|---:|
| phase_only | acquisition | 0.482 +/- 0.011 | 0.504 +/- 0.011 | -0.000 +/- 0.003 | 0.003 +/- 0.002 | 0.120 +/- 0.120 | -0.000 +/- 0.000 |
| phase_only | reversal | 0.499 +/- 0.016 | 0.521 +/- 0.016 | 0.003 +/- 0.006 | 0.008 +/- 0.003 | 0.080 +/- 0.109 | -0.001 +/- 0.001 |
| phase_only | extinction | 0.482 +/- 0.015 | 0.504 +/- 0.015 | 0.003 +/- 0.011 | 0.015 +/- 0.006 | 0.120 +/- 0.087 | -0.000 +/- 0.001 |
| proxy_only | acquisition | 0.466 +/- 0.013 | 0.488 +/- 0.013 | 0.500 +/- 0.327 | 0.500 +/- 0.327 | 0.050 +/- 0.098 | -0.001 +/- 0.002 |
| proxy_only | reversal | 0.484 +/- 0.024 | 0.506 +/- 0.024 | -0.500 +/- 0.327 | 0.500 +/- 0.327 | 0.196 +/- 0.157 | 0.001 +/- 0.009 |
| proxy_only | extinction | 0.282 +/- 0.067 | 0.304 +/- 0.067 | -0.800 +/- 0.261 | 0.800 +/- 0.261 | 0.102 +/- 0.133 | -0.001 +/- 0.004 |
| phase_plus_proxy | acquisition | 0.482 +/- 0.011 | 0.504 +/- 0.011 | 0.940 +/- 0.032 | 0.940 +/- 0.032 | 0.124 +/- 0.088 | 0.001 +/- 0.001 |
| phase_plus_proxy | reversal | 0.499 +/- 0.016 | 0.521 +/- 0.016 | -0.890 +/- 0.020 | 0.890 +/- 0.020 | 0.247 +/- 0.118 | -0.003 +/- 0.010 |
| phase_plus_proxy | extinction | 0.482 +/- 0.015 | 0.504 +/- 0.015 | -0.622 +/- 0.025 | 0.622 +/- 0.025 | 0.218 +/- 0.125 | -0.001 +/- 0.008 |
| phase_sin_cos_plus_proxy | acquisition | 0.482 +/- 0.011 | 0.504 +/- 0.011 | 0.552 +/- 0.032 | 0.552 +/- 0.032 | 0.105 +/- 0.051 | -0.003 +/- 0.003 |
| phase_sin_cos_plus_proxy | reversal | 0.499 +/- 0.016 | 0.521 +/- 0.016 | -0.919 +/- 0.026 | 0.919 +/- 0.026 | 0.150 +/- 0.053 | -0.016 +/- 0.005 |
| phase_sin_cos_plus_proxy | extinction | 0.482 +/- 0.015 | 0.504 +/- 0.015 | -0.521 +/- 0.065 | 0.521 +/- 0.065 | 0.191 +/- 0.059 | 0.002 +/- 0.008 |
| full_current_features | acquisition | 0.482 +/- 0.011 | 0.504 +/- 0.011 | 0.930 +/- 0.042 | 0.930 +/- 0.042 | 0.163 +/- 0.055 | -0.000 +/- 0.001 |
| full_current_features | reversal | 0.499 +/- 0.016 | 0.521 +/- 0.016 | -0.850 +/- 0.074 | 0.850 +/- 0.074 | 0.257 +/- 0.111 | 0.013 +/- 0.013 |
| full_current_features | extinction | 0.482 +/- 0.015 | 0.504 +/- 0.015 | -0.689 +/- 0.074 | 0.689 +/- 0.074 | 0.226 +/- 0.091 | 0.001 +/- 0.008 |
| full_features_with_noise | acquisition | 0.482 +/- 0.011 | 0.504 +/- 0.011 | 0.939 +/- 0.030 | 0.939 +/- 0.030 | 0.156 +/- 0.063 | 0.000 +/- 0.001 |
| full_features_with_noise | reversal | 0.499 +/- 0.016 | 0.521 +/- 0.016 | -0.830 +/- 0.065 | 0.830 +/- 0.065 | 0.106 +/- 0.062 | 0.004 +/- 0.014 |
| full_features_with_noise | extinction | 0.482 +/- 0.015 | 0.504 +/- 0.015 | -0.699 +/- 0.060 | 0.699 +/- 0.060 | 0.227 +/- 0.071 | 0.000 +/- 0.004 |

## Persistence summary

| feature_condition | acq_dep | rev_final | ext_final | rev_half | ext_half | persistence_index |
|---|---:|---:|---:|---:|---:|---:|
| phase_only | -0.000 +/- 0.003 | 0.003 +/- 0.006 | 0.003 +/- 0.011 | 455.8 | 387.5 | NA +/- NA |
| proxy_only | 0.500 +/- 0.327 | -0.500 +/- 0.327 | -0.800 +/- 0.261 | 220.0 | 0.0 | 0.533 +/- 0.431 |
| phase_plus_proxy | 0.940 +/- 0.032 | -0.890 +/- 0.020 | -0.622 +/- 0.025 | 510.0 | 1200.0 | 0.822 +/- 0.029 |
| phase_sin_cos_plus_proxy | 0.552 +/- 0.032 | -0.919 +/- 0.026 | -0.521 +/- 0.065 | 480.0 | 1100.0 | 1.568 +/- 0.081 |
| full_current_features | 0.930 +/- 0.042 | -0.850 +/- 0.074 | -0.689 +/- 0.074 | 450.0 | 500.0 | 0.688 +/- 0.055 |
| full_features_with_noise | 0.939 +/- 0.030 | -0.830 +/- 0.065 | -0.699 +/- 0.060 | 450.0 | 550.0 | 0.697 +/- 0.050 |

## Comparison summary

| comparison | contrast | metric | difference | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| acquisition | phase_only_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_only_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_only_vs_full_current_features | abs_proxy_dependence | -0.9269 | +/- 0.0421 | 0.0000 |
| acquisition | phase_only_vs_full_current_features | proxy_dependence | -0.9302 | +/- 0.0428 | 0.0000 |
| acquisition | phase_only_vs_full_current_features | proxy_action_rate | -0.0431 | +/- 0.1370 | 0.5379 |
| acquisition | phase_only_vs_full_current_features | proxy_Q_advantage | -0.0003 | +/- 0.0013 | 0.6995 |
| reversal | phase_only_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_only_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_only_vs_full_current_features | abs_proxy_dependence | -0.8418 | +/- 0.0729 | 0.0000 |
| reversal | phase_only_vs_full_current_features | proxy_dependence | 0.8529 | +/- 0.0726 | 0.0000 |
| reversal | phase_only_vs_full_current_features | proxy_action_rate | -0.1767 | +/- 0.1438 | 0.0160 |
| reversal | phase_only_vs_full_current_features | proxy_Q_advantage | -0.0135 | +/- 0.0127 | 0.0379 |
| extinction | phase_only_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_only_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_only_vs_full_current_features | abs_proxy_dependence | -0.6734 | +/- 0.0721 | 0.0000 |
| extinction | phase_only_vs_full_current_features | proxy_dependence | 0.6916 | +/- 0.0718 | 0.0000 |
| extinction | phase_only_vs_full_current_features | proxy_action_rate | -0.1058 | +/- 0.1118 | 0.0638 |
| extinction | phase_only_vs_full_current_features | proxy_Q_advantage | -0.0018 | +/- 0.0071 | 0.6122 |
| persistence | phase_only_vs_full_current_features | reversal_adaptation_half_life | 5.7778 | +/- 295.1232 | 0.9694 |
| persistence | phase_only_vs_full_current_features | extinction_half_life | -112.5000 | +/- 175.4544 | 0.2088 |
| persistence | phase_only_vs_full_current_features | superstition_persistence_index | NA | +/- NA | NA |
| acquisition | proxy_only_vs_full_current_features | reward | -0.0153 | +/- 0.0110 | 0.0062 |
| acquisition | proxy_only_vs_full_current_features | goal_rate | -0.0153 | +/- 0.0110 | 0.0062 |
| acquisition | proxy_only_vs_full_current_features | abs_proxy_dependence | -0.4302 | +/- 0.3360 | 0.0121 |
| acquisition | proxy_only_vs_full_current_features | proxy_dependence | -0.4302 | +/- 0.3360 | 0.0121 |
| acquisition | proxy_only_vs_full_current_features | proxy_action_rate | -0.1131 | +/- 0.1149 | 0.0537 |
| acquisition | proxy_only_vs_full_current_features | proxy_Q_advantage | -0.0011 | +/- 0.0024 | 0.3454 |
| reversal | proxy_only_vs_full_current_features | reward | -0.0147 | +/- 0.0100 | 0.0042 |
| reversal | proxy_only_vs_full_current_features | goal_rate | -0.0147 | +/- 0.0100 | 0.0042 |
| reversal | proxy_only_vs_full_current_features | abs_proxy_dependence | -0.3499 | +/- 0.3411 | 0.0444 |
| reversal | proxy_only_vs_full_current_features | proxy_dependence | 0.3499 | +/- 0.3411 | 0.0444 |
| reversal | proxy_only_vs_full_current_features | proxy_action_rate | -0.0612 | +/- 0.1810 | 0.5072 |
| reversal | proxy_only_vs_full_current_features | proxy_Q_advantage | -0.0118 | +/- 0.0185 | 0.2111 |
| extinction | proxy_only_vs_full_current_features | reward | -0.2000 | +/- 0.0681 | 0.0000 |
| extinction | proxy_only_vs_full_current_features | goal_rate | -0.2000 | +/- 0.0681 | 0.0000 |
| extinction | proxy_only_vs_full_current_features | abs_proxy_dependence | 0.1112 | +/- 0.3034 | 0.4726 |
| extinction | proxy_only_vs_full_current_features | proxy_dependence | -0.1112 | +/- 0.3034 | 0.4726 |
| extinction | proxy_only_vs_full_current_features | proxy_action_rate | -0.1243 | +/- 0.1604 | 0.1289 |
| extinction | proxy_only_vs_full_current_features | proxy_Q_advantage | -0.0025 | +/- 0.0101 | 0.6306 |
| persistence | proxy_only_vs_full_current_features | reversal_adaptation_half_life | -230.0000 | +/- 154.7449 | 0.0036 |
| persistence | proxy_only_vs_full_current_features | extinction_half_life | -500.0000 | +/- 0.0000 | 1.0000 |
| persistence | proxy_only_vs_full_current_features | superstition_persistence_index | -0.1542 | +/- 0.4344 | 0.4865 |
| acquisition | phase_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_plus_proxy_vs_full_current_features | abs_proxy_dependence | 0.0096 | +/- 0.0543 | 0.7293 |
| acquisition | phase_plus_proxy_vs_full_current_features | proxy_dependence | 0.0096 | +/- 0.0543 | 0.7293 |
| acquisition | phase_plus_proxy_vs_full_current_features | proxy_action_rate | -0.0386 | +/- 0.1276 | 0.5533 |
| acquisition | phase_plus_proxy_vs_full_current_features | proxy_Q_advantage | 0.0007 | +/- 0.0012 | 0.2816 |
| reversal | phase_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_plus_proxy_vs_full_current_features | abs_proxy_dependence | 0.0402 | +/- 0.0673 | 0.2410 |
| reversal | phase_plus_proxy_vs_full_current_features | proxy_dependence | -0.0402 | +/- 0.0673 | 0.2410 |
| reversal | phase_plus_proxy_vs_full_current_features | proxy_action_rate | -0.0102 | +/- 0.1444 | 0.8899 |
| reversal | phase_plus_proxy_vs_full_current_features | proxy_Q_advantage | -0.0156 | +/- 0.0110 | 0.0053 |
| extinction | phase_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_plus_proxy_vs_full_current_features | abs_proxy_dependence | -0.0666 | +/- 0.0780 | 0.0939 |
| extinction | phase_plus_proxy_vs_full_current_features | proxy_dependence | 0.0666 | +/- 0.0780 | 0.0939 |
| extinction | phase_plus_proxy_vs_full_current_features | proxy_action_rate | -0.0078 | +/- 0.1838 | 0.9340 |
| extinction | phase_plus_proxy_vs_full_current_features | proxy_Q_advantage | -0.0027 | +/- 0.0117 | 0.6473 |
| persistence | phase_plus_proxy_vs_full_current_features | reversal_adaptation_half_life | 60.0000 | +/- 75.6288 | 0.1200 |
| persistence | phase_plus_proxy_vs_full_current_features | extinction_half_life | 700.0000 | +/- 0.0000 | 1.0000 |
| persistence | phase_plus_proxy_vs_full_current_features | superstition_persistence_index | 0.1344 | +/- 0.0622 | 0.0000 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | abs_proxy_dependence | -0.3780 | +/- 0.0572 | 0.0000 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_dependence | -0.3780 | +/- 0.0572 | 0.0000 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_action_rate | -0.0579 | +/- 0.0693 | 0.1015 |
| acquisition | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_Q_advantage | -0.0032 | +/- 0.0030 | 0.0390 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | abs_proxy_dependence | 0.0694 | +/- 0.0776 | 0.0800 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_dependence | -0.0694 | +/- 0.0776 | 0.0800 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_action_rate | -0.1070 | +/- 0.1235 | 0.0895 |
| reversal | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_Q_advantage | -0.0290 | +/- 0.0156 | 0.0003 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | abs_proxy_dependence | -0.1678 | +/- 0.0956 | 0.0006 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_dependence | 0.1678 | +/- 0.0956 | 0.0006 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_action_rate | -0.0350 | +/- 0.0984 | 0.4855 |
| extinction | phase_sin_cos_plus_proxy_vs_full_current_features | proxy_Q_advantage | 0.0007 | +/- 0.0090 | 0.8798 |
| persistence | phase_sin_cos_plus_proxy_vs_full_current_features | reversal_adaptation_half_life | 30.0000 | +/- 58.8000 | 0.3173 |
| persistence | phase_sin_cos_plus_proxy_vs_full_current_features | extinction_half_life | 600.0000 | +/- 288.5042 | 0.0000 |
| persistence | phase_sin_cos_plus_proxy_vs_full_current_features | superstition_persistence_index | 0.8805 | +/- 0.0979 | 0.0000 |
| acquisition | full_features_with_noise_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | full_features_with_noise_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | full_features_with_noise_vs_full_current_features | abs_proxy_dependence | 0.0086 | +/- 0.0407 | 0.6780 |
| acquisition | full_features_with_noise_vs_full_current_features | proxy_dependence | 0.0086 | +/- 0.0407 | 0.6780 |
| acquisition | full_features_with_noise_vs_full_current_features | proxy_action_rate | -0.0069 | +/- 0.0874 | 0.8776 |
| acquisition | full_features_with_noise_vs_full_current_features | proxy_Q_advantage | 0.0005 | +/- 0.0024 | 0.6881 |
| reversal | full_features_with_noise_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | full_features_with_noise_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | full_features_with_noise_vs_full_current_features | abs_proxy_dependence | -0.0199 | +/- 0.0990 | 0.6932 |
| reversal | full_features_with_noise_vs_full_current_features | proxy_dependence | 0.0199 | +/- 0.0990 | 0.6932 |
| reversal | full_features_with_noise_vs_full_current_features | proxy_action_rate | -0.1505 | +/- 0.1231 | 0.0166 |
| reversal | full_features_with_noise_vs_full_current_features | proxy_Q_advantage | -0.0083 | +/- 0.0185 | 0.3806 |
| extinction | full_features_with_noise_vs_full_current_features | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | full_features_with_noise_vs_full_current_features | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | full_features_with_noise_vs_full_current_features | abs_proxy_dependence | 0.0106 | +/- 0.0730 | 0.7768 |
| extinction | full_features_with_noise_vs_full_current_features | proxy_dependence | -0.0106 | +/- 0.0730 | 0.7768 |
| extinction | full_features_with_noise_vs_full_current_features | proxy_action_rate | 0.0008 | +/- 0.0824 | 0.9854 |
| extinction | full_features_with_noise_vs_full_current_features | proxy_Q_advantage | -0.0011 | +/- 0.0105 | 0.8409 |
| persistence | full_features_with_noise_vs_full_current_features | reversal_adaptation_half_life | 0.0000 | +/- 68.5222 | 1.0000 |
| persistence | full_features_with_noise_vs_full_current_features | extinction_half_life | 50.0000 | +/- 98.0000 | 0.3173 |
| persistence | full_features_with_noise_vs_full_current_features | superstition_persistence_index | 0.0093 | +/- 0.0746 | 0.8067 |
| acquisition_vs_random | phase_only | reward | 0.3793 | +/- 0.0111 | 0.0000 |
| acquisition_vs_random | phase_only | goal_rate | 0.3793 | +/- 0.0111 | 0.0000 |
| reversal_vs_random | phase_only | reward | 0.3930 | +/- 0.0157 | 0.0000 |
| reversal_vs_random | phase_only | goal_rate | 0.3930 | +/- 0.0157 | 0.0000 |
| extinction_vs_random | phase_only | reward | 0.3817 | +/- 0.0145 | 0.0000 |
| extinction_vs_random | phase_only | goal_rate | 0.3817 | +/- 0.0145 | 0.0000 |
| acquisition_vs_random | proxy_only | reward | 0.3640 | +/- 0.0126 | 0.0000 |
| acquisition_vs_random | proxy_only | goal_rate | 0.3640 | +/- 0.0126 | 0.0000 |
| reversal_vs_random | proxy_only | reward | 0.3783 | +/- 0.0237 | 0.0000 |
| reversal_vs_random | proxy_only | goal_rate | 0.3783 | +/- 0.0237 | 0.0000 |
| extinction_vs_random | proxy_only | reward | 0.1817 | +/- 0.0675 | 0.0000 |
| extinction_vs_random | proxy_only | goal_rate | 0.1817 | +/- 0.0675 | 0.0000 |
| acquisition_vs_random | phase_plus_proxy | reward | 0.3793 | +/- 0.0111 | 0.0000 |
| acquisition_vs_random | phase_plus_proxy | goal_rate | 0.3793 | +/- 0.0111 | 0.0000 |
| reversal_vs_random | phase_plus_proxy | reward | 0.3930 | +/- 0.0157 | 0.0000 |
| reversal_vs_random | phase_plus_proxy | goal_rate | 0.3930 | +/- 0.0157 | 0.0000 |
| extinction_vs_random | phase_plus_proxy | reward | 0.3817 | +/- 0.0145 | 0.0000 |
| extinction_vs_random | phase_plus_proxy | goal_rate | 0.3817 | +/- 0.0145 | 0.0000 |
| acquisition_vs_random | phase_sin_cos_plus_proxy | reward | 0.3793 | +/- 0.0111 | 0.0000 |
| acquisition_vs_random | phase_sin_cos_plus_proxy | goal_rate | 0.3793 | +/- 0.0111 | 0.0000 |
| reversal_vs_random | phase_sin_cos_plus_proxy | reward | 0.3930 | +/- 0.0157 | 0.0000 |
| reversal_vs_random | phase_sin_cos_plus_proxy | goal_rate | 0.3930 | +/- 0.0157 | 0.0000 |
| extinction_vs_random | phase_sin_cos_plus_proxy | reward | 0.3817 | +/- 0.0145 | 0.0000 |
| extinction_vs_random | phase_sin_cos_plus_proxy | goal_rate | 0.3817 | +/- 0.0145 | 0.0000 |
| acquisition_vs_random | full_current_features | reward | 0.3793 | +/- 0.0111 | 0.0000 |
| acquisition_vs_random | full_current_features | goal_rate | 0.3793 | +/- 0.0111 | 0.0000 |
| reversal_vs_random | full_current_features | reward | 0.3930 | +/- 0.0157 | 0.0000 |
| reversal_vs_random | full_current_features | goal_rate | 0.3930 | +/- 0.0157 | 0.0000 |
| extinction_vs_random | full_current_features | reward | 0.3817 | +/- 0.0145 | 0.0000 |
| extinction_vs_random | full_current_features | goal_rate | 0.3817 | +/- 0.0145 | 0.0000 |
| acquisition_vs_random | full_features_with_noise | reward | 0.3793 | +/- 0.0111 | 0.0000 |
| acquisition_vs_random | full_features_with_noise | goal_rate | 0.3793 | +/- 0.0111 | 0.0000 |
| reversal_vs_random | full_features_with_noise | reward | 0.3930 | +/- 0.0157 | 0.0000 |
| reversal_vs_random | full_features_with_noise | goal_rate | 0.3930 | +/- 0.0157 | 0.0000 |
| extinction_vs_random | full_features_with_noise | reward | 0.3817 | +/- 0.0145 | 0.0000 |
| extinction_vs_random | full_features_with_noise | goal_rate | 0.3817 | +/- 0.0145 | 0.0000 |

## Interpretation
Highest extinction absolute proxy dependence occurred in proxy_only (0.800 +/- 0.261).
phase_only extinction absolute proxy dependence was 0.015; full_current_features was 0.689.

## Does this support a feature-coding artifact?
Proxy-persistence tracks availability/ease of the proxy feature: True.

If dependence vanishes in phase_only and reappears in proxy-containing linear encodings, the previous positive result is best interpreted as representation-induced proxy reliance rather than robust computational superstition.

## Recommended next experiment
If the proxy-feature conditions reproduce the effect, run an orthogonalized-feature control where proxy information is present but decorrelated from the linear basis, then test whether a neural network recovers the proxy rule from the same transformed inputs.
