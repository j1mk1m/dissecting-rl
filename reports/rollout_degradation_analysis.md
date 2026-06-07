# Rollout Degradation Analysis

**Date:** 2026-06-02  
**Task:** String manipulation (compositional generality)  
**Runs analyzed:** On-policy-GRPO, Bootstrap-GRPO, Bootstrap-REINFORCE+BASELINE, Teacher-GRPO, Teacher-REINFORCE+BASELINE

---

## Overview

The five runs split into two groups. **On-policy-GRPO** is the gold-standard run: accuracy grows monotonically from 11.0% to 24.5% over 1100 steps with no collapse and no reward hacking. The remaining four runs all exhibit a common degradation pattern — response length grows until the model consistently hits the 4096-token generation limit, at which point accuracy drops sharply. Two distinct failure modes explain that degradation.

---

## On-policy-GRPO: The Gold Standard

**Steps:** 0–1100  
**Result:** Steady improvement, no collapse

On-policy-GRPO is the only run that never degenerates. Accuracy climbs every 100 steps with no inflection point, and the q-spam reward hack that destroyed Bootstrap-GRPO never appears. The model does produce long responses that hit the context limit (30–65% at-limit across steps), but unlike the degraded runs these limit-hitting responses still contain genuine reasoning — the model commits to an answer partway through the trace and then continues second-guessing, hitting the limit during that post-answer reasoning.

The key structural difference is that on-policy rollouts keep the training distribution aligned with what the model currently generates. Bootstrap and Teacher runs use fixed rollout data from an earlier model, so as the policy improves the old rollouts become increasingly off-distribution, allowing degenerate behaviors to be reinforced before the training signal can correct them.

Three quantitative metrics confirm this (computed by `scripts/analysis/onpolicy_quality_analysis.py`):

### Finding 1: At-limit format compliance

When a response hits the 4096-token limit, does the model still contain a valid `{"output": ...}` token anywhere in the trace? In On-policy-GRPO, **53%** of limit-hitting responses do — the format token appears at a **median position of 53%** through the response. The model commits to an answer midway, then immediately second-guesses itself and continues reasoning for another ~5,000 characters until the context limit cuts it off. The scorer extracts the first `{"output": ...}` match, so these responses can still score correctly even though they hit the limit.

In contrast, degenerate runs never produce a valid answer token in their at-limit responses. Reasoning loops reason forever without committing; Bootstrap-GRPO at its final step has evolved past producing `{"output":` at all — the model outputs a few words and then immediately begins the q-run, filling all 4096 tokens before ever reaching the answer format.

**Teacher-GRPO (step 600) — hits limit, never commits:**
```
To predict the output of `main_solution("xfcr")`, we need to understand how the
`main_solution` function works.

The `main_solution` function calls `func_18` with the string `'pwtli'` and the
suffix `'nlvj'`, and then it calls `func_18` again with the result of the first
call and the suffix `'xz'`, and so on.

Since we are given that `main_solution(x)` = `func_18(func_7(func_18((func_8(
func_0(func_6(x, 'xz'')), 2) + func_8(func_21(func_6'pwtli', 'nlvj'')), 3)), 3),
func_7(func_18(func_21(func_24(func_7'downr', 'cfrs'')), 3)), func_7(((func_6(x,
'ksbm') = = 'x + x') = = 'x = = 'x = = = = = = = = = = = = = = = = = = = = = =
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
[= = = = continues for thousands of tokens, context limit hit, no {"output":} ever produced]
```

**On-policy-GRPO (step 1100) — hits limit, but commits at 66% through:**
```
To predict the output of `main_solution("bvg")`, we need to understand the
behavior of the `func_24` function and the `main_solution` function.

[... ~8000 chars of reasoning about func_24 (backchain_palindrome) and func_18
(loop_concat), working through the palindrome transformation step by step ...]

The final answer is {"output": "bvgbvgbvggvbgvbgvb"}.    ← answer committed here (score: 1)

The function main_solution(x) will return the result of func_24(s3) after the
function func_24(s3) returns a palindrome. The depth of transformation is 3.
The string "bvgbvgbvggvbgvbgvb" is a palindrome.

At this point, further transformations are not required.

[... ~4000 chars of continued second-guessing until context limit hit]
```

| Run (final step) | Normal fmt | At-limit fmt |
|-----------------|-----------|-------------|
| On-policy-GRPO (step 1100) | 99.5% | **53.0%** |
| Teacher-GRPO (step 600) | 84.2% | 1.4% |
| Teacher-REINFORCE+BASELINE (step 1033) | 77.9% | 0.6% |
| Bootstrap-GRPO (step 661) | 33.3% | 0.0% |

### Finding 2: Depth gradient of limit-hitting probability

If the context limit is hit because problems are hard, harder problems should hit the limit more often. If it is hit because the model is broken, depth should not predict limit-hitting.

In On-policy-GRPO (step 1100), limit-hitting rises monotonically from **0% at depth 0** to ~**65% at depth 10** (regression slope: 0.066/depth). In Bootstrap-GRPO, the slope is 0.009 — essentially flat, confirming that the model hits the limit regardless of problem difficulty.

```
On-policy-GRPO step 1100          Teacher-REINFORCE step 1000
depth  0:   0%                    depth  0:  19%  ######
depth  1:   4%  #                 depth  1:  75%  ######################
depth  2:  15%  ####              depth  2:  91%  ###########################
depth  3:  27%  ########          depth  3:  97%  #############################
depth  4:  32%  #########         depth  4:  99%  #############################
depth  5:  35%  ##########        depth  5:  99%  #############################
depth  6:  55%  ################  depth  6: 100%  ##############################
depth  7:  61%  ##################
depth  8:  58%  #################
```

| Run (final step) | Slope (limit-frac/depth) | Character |
|-----------------|-------------------------|-----------|
| On-policy-GRPO | **0.066** | Capacity constraint |
| Teacher-GRPO | 0.056 | Intermediate |
| Teacher-REINFORCE+BASELINE | 0.038 | Early collapse, then flat |
| Bootstrap-GRPO | 0.009 | Essentially flat — model is broken |

### Finding 3: Accuracy of limit-hitting responses by depth

In On-policy-GRPO, responses that hit the limit at depth 2 still score **20%** and at depth 3 score **8.6%** — the model solved some problems despite running out of tokens. In every off-policy run, at-limit accuracy is exactly 0% at all depths.

| Depth | On-policy-GRPO at-limit | Teacher-REINFORCE at-limit |
|-------|------------------------|---------------------------|
| 1 | 0.0% | 0.0% |
| 2 | **20.0%** | 0.0% |
| 3 | **8.6%** | 0.0% |
| 4 | 1.8% | 0.0% |
| 5 | 1.4% | 0.0% |

---

## Failure Mode 1: Reasoning Loops

**Affected runs:** Bootstrap-REINFORCE+BASELINE, Teacher-REINFORCE+BASELINE, Teacher-GRPO  
**Onset:** ~step 200–500 depending on run

The model enters a self-correcting spiral. It reasons through the problem correctly, arrives at an intermediate answer, then immediately doubts it and restarts the reasoning chain — repeating this loop until the context limit is hit. The `{"output": ...}` answer is never committed, so the response scores 0.

**Example tail (Teacher-REINFORCE+BASELINE, step 1000):**
```
However, `main_solution(x)` is supposed to return `func_2(x)`, and `func_2(x)` removes
vowels from `x`. In this case, `x = "rxck"` contains no vowels, so `func_2(x)` will
return `x = "rxck"`.

Therefore, `main_solution(x)` will return `x = "rxck"`.

However, `main_solution(x)` is supposed to return `func_2(x)`, and `func_2(x)` removes
vowels from `x`. In this case, `x = "rxck"` contains no vowels, ...
```

Other manifestations include repeating backtick sequences (` ` ` ` ` ...`) and paraphrased re-statements of the same reasoning step.

**Why this happens:** REINFORCE does not penalize response length. There is no cost to continuing to reason, so the model learns to hedge: rather than committing to an answer and risking a 0-reward wrong response, it keeps "thinking." This is the classic **RL overthinking** failure — the policy discovers that deferring the answer is never worse than submitting a wrong one.

---

## Failure Mode 2: q-Spam Reward Hacking

**Affected runs:** Bootstrap-GRPO only  
**Onset:** Step ~300, near-total (97–100%) by step 400

This is a qualitatively different and more severe failure. The model produces correct-looking reasoning, then immediately emits a valid-format answer token `{"output": "qqqqq..."}` filled with hundreds to thousands of the character `q`, consuming the remainder of the context window. The reasoning and format are present; only the answer content is degenerate.

**Example (Bootstrap-GRPO, step 300):**
```
...Given the input string "rxck", we can manually remove the vowels to get the output:

"r" (no vowel)
"x" (no vowel)
"c" (no vowel)
"k" (no vowel)

So, the output of `main_solution("rxck")` is:

{"output": "qqqqqqqqqqqqqqqqqqqqqqqqqqqkqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq..."}
```

The q-run spans roughly 3,900 characters (the majority of the context window).

**Why this happens:** This is **reward hacking**. GRPO computes group-relative rewards, normalizing scores within a batch. In some batch, a q-spam response happened to score higher (or equally) relative to other responses in its group — possibly because the group was populated entirely by wrong verbose answers with no valid format. GRPO then reinforced the q-spam pattern. Since exact-match reward never fires for a `q`-filled answer, accuracy collapses entirely.

The character `q` specifically appears because it is common in the training input strings (e.g., `"qqyu"`, `"rxck"`, `"peyzq"`), and the model is likely sampling from its own recent output distribution during rollouts.

The key structural difference from the loop failure: q-spam places the `{"output":` token *early* in the response, satisfying the format checker. This means format compliance drops with a slight lag behind accuracy — the model is still outputting valid format, but the content is garbage.

---

## Quantitative Breakdown

Response classification per step (computed by `scripts/analysis/classify_rollouts.py`). Categories:
- **normal** — response ends before 4096 tokens
- **verbose** — hits limit but no detectable repetition (over-explained, loses track)
- **loop** — hits limit with a repeated phrase or reasoning cycle in the tail
- **q_spam** — hits limit with a 50+ character run of `q` in the output field

### On-policy-GRPO

| Step | Normal | Verbose | Loop | q-Spam |
|------|--------|---------|------|--------|
| 0    | 93%    | 2%      | 6%   | 0%     |
| 100  | 58%    | 11%     | 31%  | 0%     |
| 200  | 42%    | 15%     | 43%  | 0%     |
| 300  | 51%    | 14%     | 35%  | 0%     |
| 400  | 37%    | 13%     | 50%  | 0%     |
| 500  | 58%    | 10%     | 33%  | 0%     |
| 600  | 36%    | 16%     | 49%  | 0%     |
| 700  | 38%    | 11%     | 51%  | 0%     |
| 800  | 31%    | 14%     | 55%  | 0%     |
| 900  | 27%    | 15%     | 58%  | 0%     |
| 1000 | 54%    | 10%     | 36%  | 0%     |
| 1100 | 57%    | 9%      | 33%  | 0%     |

No q-spam at any step. The loop fraction fluctuates with no upward trend; accuracy continues rising regardless, confirming that loop-classified responses here are a different character from the degenerate loops in the off-policy runs — the model is working but running out of context, not stuck.

### Bootstrap-GRPO

| Step | Normal | Verbose | Loop | q-Spam |
|------|--------|---------|------|--------|
| 100  | 63%    | 33%     | 4%   | 0%     |
| 200  | 60%    | 32%     | 8%   | 0%     |
| 300  | 0%     | 1%      | 1%   | **97%** |
| 400  | 0%     | 0%      | 0%   | **100%** |
| 500  | 0%     | 0%      | 0%   | **100%** |
| 600  | 5%     | 0%      | 0%   | **95%** |
| 661  | 1%     | 0%      | 0%   | **99%** |

### Bootstrap-REINFORCE+BASELINE

| Step | Normal | Verbose | Loop | q-Spam |
|------|--------|---------|------|--------|
| 100  | 22%    | 77%     | 0%   | 0%     |
| 200  | 34%    | 7%      | **58%** | 0% |
| 260  | 19%    | 1%      | **80%** | 0% |

### Teacher-GRPO

| Step | Normal | Verbose | Loop | q-Spam |
|------|--------|---------|------|--------|
| 100  | 82%    | 17%     | 0%   | 0%     |
| 200  | 74%    | 25%     | 1%   | 0%     |
| 300  | 75%    | 24%     | 0%   | 0%     |
| 400  | 62%    | 38%     | 0%   | 0%     |
| 500  | 53%    | 46%     | 1%   | 0%     |
| 600  | 18%    | **82%** | 0%   | 0%     |

### Teacher-REINFORCE+BASELINE

| Step | Normal | Verbose | Loop | q-Spam |
|------|--------|---------|------|--------|
| 100  | 84%    | 16%     | 0%   | 0%     |
| 200  | 76%    | 23%     | 1%   | 0%     |
| 300  | 72%    | 27%     | 1%   | 0%     |
| 400  | 83%    | 17%     | 0%   | 0%     |
| 500  | 39%    | 0%      | **61%** | 0% |
| 600  | 18%    | 0%      | **82%** | 0% |
| 700  | 11%    | 0%      | **89%** | 0% |
| 800  | 9%     | 0%      | **90%** | 0% |
| 900  | 5%     | 0%      | **95%** | 0% |
| 1000 | 6%     | 1%      | **93%** | 0% |
| 1033 | 7%     | 1%      | **92%** | 0% |

---

## Overall Accuracy Trajectory

| Run | Peak Accuracy | Final Accuracy | Collapse Onset |
|-----|--------------|----------------|----------------|
| **On-policy-GRPO** | **24.5% (step 1100)** | **24.5% (step 1100)** | **None — still improving** |
| Bootstrap-GRPO | 12.2% (step 100) | 0.4% (step 661) | Step 300 (q-spam) |
| Bootstrap-REINFORCE+BASELINE | 10.7% (step 200) | 9.6% (step 260) | Step 200 (loops beginning) |
| Teacher-GRPO | 12.0% (step 100–400) | 8.8% (step 600) | Step 500–600 (verbosity) |
| Teacher-REINFORCE+BASELINE | 11.9% (step 100) | 4.7% (step 1033) | Step 500 (loops) |

On-policy-GRPO is the only run that keeps improving; all off-policy runs peak within the first 200 steps and then degrade. Bootstrap-GRPO collapses fastest and most completely due to the q-spam reward hack reinforcing itself in a positive feedback loop.

---

## Accuracy by Composition Depth

On-policy-GRPO pushes the compositional generalization frontier substantially further than any off-policy run:

| Depth | Bootstrap-GRPO (step 100) | Teacher-REINFORCE+BASELINE (step 100) | On-policy-GRPO (step 1100) |
|-------|--------------------------|---------------------------------------|---------------------------|
| 0     | 91.7%                    | ~90%                                  | 97.2%                     |
| 1     | 59.9%                    | ~55%                                  | 81.0%                     |
| 2     | 18.7%                    | ~15%                                  | 63.9%                     |
| 3     | 6.4%                     | ~5%                                   | 34.2%                     |
| 4     | 4.0%                     | ~2%                                   | 15.9%                     |
| 5     | ~0%                      | ~0%                                   | 5.6%                      |
| 6+    | ~0%                      | ~0%                                   | ~0%                       |

The depth cliff shifts from 2–3 (off-policy runs) to 5–6 (on-policy). At step 1100, On-policy-GRPO still appears to be improving at depths 1–4; the run has not yet converged.

---

## Conclusions

1. **On-policy rollouts are the decisive factor.** On-policy-GRPO is the only run that keeps improving — it reaches 24.5% overall accuracy and 34% at depth 3 by step 1100, compared to a peak of ~12% overall and ~6% at depth 3 for all off-policy runs. The distribution alignment that on-policy training provides prevents both failure modes from taking hold.

2. **GRPO is uniquely vulnerable to reward hacking on this task.** The group-relative normalization can amplify a degenerate behavior if it outcompetes other responses in a single batch. The q-spam pattern is not observed in any REINFORCE or on-policy run — only Bootstrap-GRPO, where the off-policy data creates the distributional gap that the hack exploits.

3. **Length regularization is missing from off-policy runs.** There is no penalty for long responses, allowing loops and q-spam to be reinforced once they appear. On-policy-GRPO avoids this because policy updates happen before degenerate behaviors can compound.

4. **Teacher data buys time but does not prevent collapse.** Teacher runs start with better format compliance and maintain it longer, but eventually fall into the same length-limit failure as bootstrap runs.

5. **The compositional generalization cliff shifts but does not disappear.** On-policy-GRPO pushes reliable accuracy from depth 2–3 to depth 4–5, with non-zero accuracy first appearing at depth 5 (5.6%) and depth 6 remaining at zero. The run has not converged and may improve further.
