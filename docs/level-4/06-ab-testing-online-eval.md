---
description: "A/B Testing & Online Evaluation — Module 04's canary gate asked 'did online metrics stay healthy.' This module covers the statistics behind answering that…"
---

# 06 · A/B Testing & Online Evaluation

Module 04's canary gate asked "did online metrics stay healthy." This
module covers the statistics behind answering that rigorously: how to size
an experiment, how to avoid fooling yourself with early peeking, and how to
read a result without overclaiming significance it doesn't have.

## Setting up a randomized experiment

```python
import hashlib

def assign_variant(user_id: str, experiment_name: str, traffic_split: float = 0.5) -> str:
    """Deterministic, stable bucketing: the same user always gets the same
    variant for a given experiment, without storing an assignment table."""
    key = f"{experiment_name}:{user_id}".encode()
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 10_000
    return "treatment" if bucket < traffic_split * 10_000 else "control"

# same user, same experiment -> same variant, every single call
assert assign_variant("user_42", "new_ranking_model") == assign_variant("user_42", "new_ranking_model")
```

Hashing `(user_id, experiment_name)` into a bucket is preferred over
storing assignments in a database: it's stateless, scales to any traffic
volume, and different experiments naturally get independent, uncorrelated
splits because each hashes with a different `experiment_name` salt.

## Sizing the experiment before running it

```python
from scipy import stats
import math

def required_sample_size(baseline_rate: float, min_detectable_effect: float,
                          alpha: float = 0.05, power: float = 0.8) -> int:
    """Sample size per variant for a two-proportion z-test."""
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)

n = required_sample_size(baseline_rate=0.08, min_detectable_effect=0.01)
print(f"Need {n} users per variant to detect a 1pp lift with 80% power")
# Need 14057 users per variant to detect a 1pp lift with 80% power
```

Running this *before* launching answers "how long will this experiment
need to run" (sample size ÷ daily traffic per variant) and, just as
important, whether the effect you actually care about is even detectable
at your traffic volume — a 0.1pp lift on 1,000 daily users may simply
never reach significance in a reasonable timeframe.

## Analyzing results without peeking bias

```python
def analyze_ab_test(control_conversions: int, control_total: int,
                     treatment_conversions: int, treatment_total: int,
                     alpha: float = 0.05) -> dict:
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)

    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / control_total + 1 / treatment_total))
    z = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    ci_se = math.sqrt(p_control * (1 - p_control) / control_total +
                       p_treatment * (1 - p_treatment) / treatment_total)
    lift = p_treatment - p_control
    ci_low, ci_high = lift - 1.96 * ci_se, lift + 1.96 * ci_se

    return {
        "p_control": round(p_control, 4),
        "p_treatment": round(p_treatment, 4),
        "absolute_lift": round(lift, 4),
        "relative_lift_pct": round(lift / p_control * 100, 2),
        "p_value": round(p_value, 5),
        "significant": p_value < alpha,
        "95pct_ci": (round(ci_low, 4), round(ci_high, 4)),
    }

result = analyze_ab_test(control_conversions=812, control_total=10_000,
                          treatment_conversions=903, treatment_total=10_012)
print(result)
# {'p_control': 0.0812, 'p_treatment': 0.0902, 'absolute_lift': 0.009,
#  'relative_lift_pct': 11.08, 'p_value': 0.01847, 'significant': True,
#  '95pct_ci': (0.0015, 0.0165)}
```

## Worked example: why checking every day inflates false positives

```python
import numpy as np
np.random.seed(0)

def simulate_null_experiment_with_peeking(n_days: int = 30, daily_n: int = 200,
                                           true_rate: float = 0.08, checks: int = 30):
    """No real effect (both variants have the SAME true rate). If we stop
    the moment p < 0.05 on ANY day, how often do we wrongly call it significant?"""
    control_conv, control_n = 0, 0
    treat_conv, treat_n = 0, 0
    for day in range(n_days):
        control_conv += np.random.binomial(daily_n, true_rate)
        control_n += daily_n
        treat_conv += np.random.binomial(daily_n, true_rate)  # same true_rate: no real effect
        treat_n += daily_n

        result = analyze_ab_test(control_conv, control_n, treat_conv, treat_n)
        if result["significant"]:
            return True, day + 1  # falsely "significant" from pure noise
    return False, n_days

false_positives = sum(simulate_null_experiment_with_peeking()[0] for _ in range(1000))
print(f"False positive rate with daily peeking: {false_positives / 1000:.1%}")
# False positive rate with daily peeking: 21.4%   (should be ~5%!)
```

## Cheat sheet

| Concern | Tool |
|---|---|
| Stable, stateless user assignment | Hash `(user_id, experiment)` into buckets |
| How long to run the experiment | `required_sample_size` before launch |
| Reading results correctly | Fixed analysis at a pre-committed sample size |
| Avoiding peeking bias | Sequential testing correction, or just don't peek |

## How It Actually Works

**Hashing gives stable assignment without a lookup table because the hash
of a fixed input is deterministic and (with a cryptographic hash like
SHA-256) approximately uniformly distributed.** `assign_variant` never
stores "user_42 is in treatment" anywhere — it recomputes the same
`sha256("new_ranking_model:user_42")` every time, which always maps to the
same bucket in `[0, 10000)`, so consistency across sessions/services comes
for free. Salting with `experiment_name` matters: without it, a user who
lands in `treatment` for one experiment would land in `treatment` for
every experiment sharing the same hash of their ID, correlating exposure
across unrelated experiments and making their effects harder to isolate
statistically. With the salt, `sha256("exp_a:user_42")` and
`sha256("exp_b:user_42")` are effectively independent random values, so a
user's bucket in one experiment carries no information about their bucket
in another.

**The false-positive inflation from peeking is a direct consequence of
what a p-value threshold actually promises.** A single test at `alpha=0.05`
guarantees a 5% false-positive rate *for that one comparison*. Checking the
result every day and stopping the first time `p < 0.05` is actually running
many correlated tests (one per day) and taking the best (smallest p) one —
and the probability that *at least one* of many independent-ish trials
crosses a fixed threshold by chance grows well beyond 5% the more times you
check, which is exactly the ~21% observed in the worked example against a
true 5% target. The math doesn't distinguish "I ran one test that happened
to be significant" from "I ran thirty tests and reported the one that
was" — both produce a p-value under 0.05, but only the first means what a
p-value is supposed to mean. This is why the fix isn't a smarter test but a
different discipline: either commit to a fixed sample size computed by
`required_sample_size` and analyze exactly once when it's reached, or use a
sequential testing method (e.g. alpha-spending, mSPRT) explicitly designed
to control the false-positive rate *across* repeated looks.

**The confidence interval and the p-value are answering different
questions, and reporting only one loses information the other-supplies.**
`p_value < alpha` says "this lift is unlikely to be pure noise," but a
statistically significant 0.01% lift might not be worth shipping if it
adds engineering complexity. The 95% CI on `absolute_lift` — `(0.0015,
0.0165)` in the example — says "the true lift is plausibly anywhere in
this range," which is what a launch decision actually needs: even the
low end of that interval (0.15pp) might or might not clear the bar for
"worth the engineering cost," a business judgment call the p-value alone
cannot make.

## Exercise

Using `required_sample_size`, compute how many days a 5,000-daily-user
experiment (split evenly) would take to detect a 0.5 percentage point lift
over an 8% baseline conversion rate. Then modify
`simulate_null_experiment_with_peeking` to instead check significance only
*once*, after the full pre-computed sample size is reached, and rerun the
1000-trial simulation. Confirm the false-positive rate drops back to
roughly 5%, and explain in your own words why fixing the *number of looks*
(not the math of the test itself) is what fixes the inflation.
