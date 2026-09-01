# Degenerate Response Examples: Teacher/Bootstrap + GRPO/REINFORCE

**Context:** concrete example generations backing the collapse findings in
`reports/math_task_response_length_dynamics.md` and
`reports/string_task_offpolicy_collapse.md` — all from frozen-pool
(Teacher/Bootstrap) runs trained with a negative-sample loss (GRPO or
REINFORCE+BASELINE). Full dumps live in
`reports/figures/string_task_examples/*.txt`; math examples pulled directly
from `rollout_eval_data/660.jsonl` in
`/data/user_data/gyeongwk/checkpoints/math-task/math-bootstrap-GRPO-easy-Qwen3-1.7B/`.

Math collapses to **brevity** (empty `<think>` block, confident bare guess).
String collapses to **infinite repetition loops** that never reach a JSON
answer. Same root cause (frozen data + advantage-weighted loss with no live
resampling to self-correct), opposite surface symptom.

---

## Math (Bootstrap-GRPO, `math-easy`, step 660)

**Example 1** — drops a term but states the (wrong) answer with total confidence:

> **Prompt:** Simplify $(2-3z) - (3+4z)$.
>
> **Output:**
> ```
> <think>
>
> </think>
>
> We are given the expression:
>
> $$
> (2 - 3z) - (3 + 4z)
> $$
>
> ### Step 1: Distribute the negative sign
>
> We distribute the negative sign to both terms inside the second parenthesis:
>
> $$
> (2 - 3z) - 3 - 4z
> $$
>
> ### Step 2: Combine like terms
>
> Group the constant terms and the terms with $ z $:
>
> - Constant terms: $ 2 - 3 = -1 $
> - $ z $ terms: $ -3z - 4z = -7z $
>
> ### Final Answer:
>
> $$
> \boxed{-7z}
> ```
> The constant term `-1` is computed correctly in the scratch line but then
> silently dropped from the final boxed answer (correct answer: `-1 - 7z`).

**Example 2** — invents a wrong unit-conversion factor with zero visible reasoning to catch it:

> **Prompt:** ...if his waist size in inches is 34 inches, what size should he
> specify in centimeters?
>
> **Output:**
> ```
> <think>
>
> </think>
>
> To determine the waist size in centimeters for Daniel, we are given:
>
> - 1 foot = 12 inches
> - 1 foot = 30.5 centimeters
> - Daniel's waist size in inches = 34 inches
>
> ### Step 1: Convert 34 inches to centimeters
>
> We know that 1 foot = 30.5 cm, so:
>
> 34 inches = 34 × 30.5 cm = 1037 cm
>
> ### Final Answer:
>
> \boxed{1037.0}
> ```
> Applies the foot→cm conversion factor directly to an inch value (conflates
> the two units) — a mistake a genuine reasoning trace would likely have
> caught.

Both examples share the same signature: `<think>\n\n</think>` (fully emptied
reasoning block) followed by a short, fluent, unverified derivation — this
pattern accounts for the large majority of responses by step 660, vs. full
genuine derivations at step 100.

---

## String task (four loop styles, all on the same prompt family)

All four runs live in `/data/user_data/gyeongwk/checkpoints/string-task/`.
Prompt used below: `main_solution("hqbgzciw")` /
`func_16(func_8((func_16(func_24(x, 1)) + func_6(func_8('agk', 3), 'ymlm')), 2), 'tli')`
(and the related `main_solution("kpur")` / `func_24('csby', 1)` prompt for
example 3), unless noted.

**1. Character-level filler** — Bootstrap-GRPO, step 300+ (99.5% of responses
capped at 4096 tokens by this point):
```
To predict the output of `main_solution("kqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq...
```
A single repeated character runs to the token cap; never reaches an answer.

**2. Symbol-loop after a genuine start** — Teacher-GRPO, step 400/600.
Reasoning is coherent for several paragraphs (correctly traces the
`func_8`/`func_6`/`func_16` call chain), settles on the same wrong conclusion
`{"output": "kagymlm"}` several times in a row, then degrades into pure
filler for the rest of the budget:
```
...func_24 is called with x = "hqbgzciw" in the main_solution function, and
func_24 returns the result of calling func_8 with s = 'agk' and n = 3. We
have determined that func_8 rotates the string 'agk' by 3 positions to the
right, resulting in 'kag'.

Therefore, the output of main_solution("hqbgzciw") will be the result of
calling func_6 with s = 'kag' and suf = 'ymlm', which is 'kag' + 'ymlm' =
'kagymlm'.

The final answer is:

{"output": "kagymlm"}

However, we need to consider the main_solution function again, which calls
func_24 with x = "hqbgzciw" and depth = 1. We have determined that func_24
will rotate the string "hqbgzciw" by 1 position to the right, resulting in
"qbgzciwh". But this is not the case, because func_24 is called with
s = func_8('agk', 3) and suf = 'ymlm', not with x = "hqbgzciw".
... (repeats ~10x verbatim, then degenerates into "= = = = = = ..." filler
    for the remainder of the 4096-token budget)
```

**3. Palindrome-recursion sentence loop** — Teacher-REINFORCE+BASELINE, step
1033 (end of the run, 93% of responses capped). Gets stuck restating the
exact same conditional sentence verbatim on the `func_24`/palindrome prompt:
```
If `s` is not a palindrome, `func_24` will return `s` if `s` is a palindrome
after appending its reverse and recursively calling itself with `d - 1`.

In this case, `s` is not a palindrome, but `func_24` will try to transform
it into a palindrome.
```
repeated ~20+ times until the length cap, never emitting a JSON answer.

**4. Circular self-contradiction loop, present even at step 100** —
Teacher-GRPO, step 100 (mixed in with normal responses from the very first
checkpoint — this pattern isn't something that only emerges after many steps
of drift):
```
...func_24 is called with s = func_8('agk', 3) and suf = 'ymlm', not with
x = "hqbgzciw".

However, we can see that func_24 is called with x = "hqbgzciw" in the
main_solution function, and func_24 returns the result of calling func_8
with s = 'agk' and n = 3. We have determined that func_8 rotates the string
'agk' by 3 positions to the right, resulting in 'kag'.

Therefore, the output of main_solution("hqbgzciw") will be the result of
calling func_6 with s = 'kag' and suf = 'ymlm', which is 'kag' + 'ymlm' =
'kagymlm'.

The final answer is:

{"output": "kagymlm"}
```
repeats this exact "However... Therefore... final answer... {"output":
"kagymlm"}" cycle roughly 10 times verbatim before hitting the token cap,
without ever letting the repeated `{"output": ...}` line actually count as
the terminal answer.

---

## Summary

| Run | Task | Failure mode |
|---|---|---|
| Bootstrap-GRPO (step 660) | math | Empty `<think>` block, confident short derivation, dropped/confused a step |
| Bootstrap-GRPO (step 300+) | string | Character-level repetition filler to the token cap |
| Teacher-GRPO (step 400/600) | string | Genuine reasoning → repeated wrong answer → symbol-level (`= = = =`) filler |
| Teacher-REINFORCE+BASELINE (step 1033) | string | Verbatim sentence-level loop, never reaches JSON |
| Teacher-GRPO (step 100) | string | Circular "However/Therefore" paragraph loop, present from the earliest checkpoint |
