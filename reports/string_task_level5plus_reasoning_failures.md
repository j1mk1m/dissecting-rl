# Level 5+ String Task Accuracy Collapse: Token Limit vs. Reasoning Failure

**Context:** `results/main.csv` shows on-policy accuracy dropping from ~70-88%
at level 2 to essentially 0 by level 5+ across every loss function (GRPO,
REINFORCE+BASELINE, POS+NEG, SFT). This report scores and tokenizes every
generation at depth 5-8 for the three eval runs with saved rollouts
(`eval/passk/{On-policy-GRPO,On-policy-SFT,stage1-rft}/rollout.parquet`, 128
samples x 256 prompts x 4 depths each) to determine whether the collapse is
an eval-harness artifact (hitting the 4096-token generation cap before
reaching an answer) or a genuine compositional-generalization failure.
Scoring uses the real reward function
(`verl/utils/reward_score/codeio.py::compute_score`); token counts use the
`gyeongwk/stage1-rft` tokenizer.

## Aggregate stats

| Run | Depth | avg tokens | % hit 4096 cap | % correct | % valid JSON but wrong | % failed to extract JSON | of those, % truncated |
|---|---|---|---|---|---|---|---|
| On-policy-GRPO | 5 | 2126 | 1.8% | 3.12% | 87.3% | 9.6% | 17.7% |
| On-policy-GRPO | 6 | 2341 | 1.5% | 0.59% | 86.6% | 12.8% | 11.0% |
| On-policy-GRPO | 7 | 2536 | 1.9% | 0.08% | 83.6% | 16.3% | 11.1% |
| On-policy-GRPO | 8 | 2543 | 2.2% | 0.05% | 82.8% | 17.1% | 12.6% |
| On-policy-SFT | 5 | 197 | 0.4% | 0.91% | 98.5% | 0.6% | 4.9% |
| On-policy-SFT | 6 | 241 | 0.3% | 0.11% | 98.8% | 1.1% | 6.2% |
| On-policy-SFT | 7 | 302 | 0.5% | 0.05% | 98.1% | 1.8% | 8.8% |
| On-policy-SFT | 8 | 313 | 0.6% | 0.00% | 97.8% | 2.2% | 10.3% |
| stage1-rft | 5 | 624 | 1.3% | 0.09% | 96.6% | 3.3% | 35.2% |
| stage1-rft | 6 | 641 | 1.3% | 0.07% | 96.7% | 3.2% | 35.2% |
| stage1-rft | 7 | 672 | 1.5% | 0.00% | 96.6% | 3.4% | 38.2% |
| stage1-rft | 8 | 670 | 1.5% | 0.00% | 96.3% | 3.7% | 35.5% |

**Only ~1.5-2.2% of generations actually hit the 4096-token cap**, across all
runs and all depths. The other **83-98%** finish normally, produce a
well-formed `{"output": ...}` JSON, and are simply **wrong**. Even among the
minority that fail to extract *any* JSON, truncation only explains 11-18% of
those for GRPO/SFT (35-38% for the untrained `stage1-rft` base checkpoint,
which loops more).

**Conclusion: the level 5+ collapse is a genuine compositional reasoning
failure, not a token-budget artifact.** Bumping `max_tokens` would fix at
most ~2% of failures. The model can correctly identify which primitive
functions are being composed and in what nesting order, but loses track of
intermediate string state as composition depth grows, and small per-step
errors compound until the final answer is unrelated to the ground truth.

---

## Example A (depth 5) — wrong rotation direction + a silently dropped step

`main_solution("gkuwjazhrx")` calls
`func_6(func_21(func_10(func_8(func_24('ckzny', 2), 3))), 'qwk')`.

Ground truth: `nYyNzKcCkZqwk`. Model output: `znyynzkcckqwk`.

Full model response (excerpted for the composed steps):

> `func_24('ckzny', 2)` will first check if `s` is a palindrome... Therefore,
> `func_24('ckzny', 2)` will append the reversed string of `'ckzny'` to
> `'ckzny'`... `'ckzny'` + `'ynzkc'` equals `'ckznyynzkc'`.
>
> ...`func_8(s, n)` rotates the string `s` by `n` positions... This means it
> moves the last `n` characters to the front of the string... Shifting these
> characters 3 positions to the right, we get: `z`, `n`, `y`, `y`, `n`, `z`,
> `k`, `c`, `c`, `k`. Therefore, `func_8(...)` will return the rotated string
> `'znyynzkcck'`.
>
> ...`func_21('znyynzkcck')` will return the string `'znyynzkcck'` itself, as
> it only contains alphabets.
>
> Finally, `main_solution(x)` returns `'znyynzkcck'qwk`.
>
> The final answer is: {"output": "znyynzkcckqwk"}

**What actually happened, step by step (verified against
`scripts/data_preprocess/string_data.py`):**

1. `func_24` (`backchain_palindrome`) recursion: the model's stated plan is
   wrong (it says it needs "2 more" rounds after the first append), but
   `'ckzny'+'ynzkc' = 'ckznyynzkc'` is already a palindrome, so the real
   recursion terminates here too — the model lands on the **correct**
   intermediate value `'ckznyynzkc'`, but by contradicting its own stated
   logic rather than by checking the palindrome-termination condition.
2. `func_8` (`rotate_str`) is defined as `s[n:] + s[:n]` — a **left**
   rotation (move the first `n` chars to the end). The model's mental model
   is backwards ("moves the last `n` characters to the front" / "shifting
   right"). Correct: `'ckznyynzkc'[3:] + 'ckznyynzkc'[:3]` =
   `'nyynzkc'+'ckz'` = `'nyynzkcckz'`. Model computed the mirror-image
   rotation instead: `'znyynzkcck'`.
3. `func_10` (`alternate_case`) is **called for in the reference code but
   never applied by the model at all** — step "12" in the response jumps
   straight from the (already wrong) `func_8` output to `func_21`, silently
   dropping an entire operation in the chain. Applying `alternate_case` to
   the correct `'nyynzkcckz'` gives `'nYyNzKcCkZ'`, exactly the ground
   truth prefix — the model never even attempts this step.
4. `func_21` (remove non-alpha) and `func_6` (add suffix `'qwk'`) are
   applied correctly (trivially, since step 4 is a no-op on
   all-alphabetic input).

Net effect: one **direction-of-rotation** error compounded with one
**silently omitted function call**, on an otherwise correctly-parsed 4-deep
composition — the output is a near-anagram of the truth (same character
multiset, similar-length rotation) rather than nonsense, which is exactly
why it "looks" close but scores 0.

---

## Example B (depth 6) — argument-scope confusion at the first nested call

`main_solution("xvic")` calls
`func_7(func_6(func_8(func_14(func_6(func_10(x), 'rev')), 1), 'cxkg'), func_16(func_16(func_7(func_6(func_12('pbjub'), 'pf'), func_6(func_8(x, 3), 'bqvl')))))`.

Ground truth: `xpVcVbixijCvC5rirbepeqvfvvxlcxkg`. Model output (two separate
samples on the same prompt): `cxxrrvveeiivvcc` and `rcvxekigvcx`.

Excerpted reasoning (second sample):

> `func_10(s)` interlaces two strings character by character... So,
> `func_6(func_10(x), 'rev')` will interlace the characters of `x` with the
> characters of `'rev'`... Interlacing `"xvic"` and `'rev'`... will produce
> the string: `'xrveivc'`.
>
> ...`func_14('xrveivc')` will produce the string: `'xxrrvveeiivvcc'`.
>
> ...rotating `'xxrrvveeiivvcc'` by 1 position [moving the last character to
> the front]... results in `'cxxrrvveeiivvcc'`.
>
> Finally, we need to find `func_7` of `'rveivcx'` and `'cxkg'`... Therefore,
> `main_solution(x)` will return the string `'rcvxekigvcx'` after calling
> `func_7`, and this is the final output.

**What actually happened:** `func_10` is `alternate_case` — a **single-
argument** function (`ch.lower() if i%2==0 else ch.upper()`); it does not
take a second string. In the reference code, `'rev'` is not an argument to
`func_10` at all — it's the *suffix* argument of the **outer** `func_6`
(`func_6(func_10(x), 'rev')` = `add_suffix(alternate_case(x), 'rev')`). The
model misattributes `'rev'` to `func_10` and additionally misremembers
`func_10`'s semantics as an interlace operation (that's actually `func_7`) —
a bracket/argument-scope error compounded with a function-identity mix-up,
both at the very first (innermost) call. Every subsequent step
(`func_14` duplication, `func_8` rotation) is then executed with locally
correct arithmetic, but on a string that was wrong from the first line.

The model also entirely **drops the second branch of the outermost
`func_7`** — the reference's second operand is a whole separate
sub-composition, `func_16(func_16(func_7(func_6(func_12('pbjub'), 'pf'),
func_6(func_8(x, 3), 'bqvl'))))`, but the model instead treats the literal
string `'cxkg'` (itself meant several levels up as an `add_suffix` argument)
as if it were directly the second operand of the outer `func_7`. This is a
more severe class of error than Example A's local rotation mistake: a
whole branch of the composition tree is discarded rather than merely
mis-computed.

---

## Takeaway

Two distinct compounding-error patterns account for the level 5+ collapse:

- **Local arithmetic/direction errors** (Example A) — the model tracks the
  right call structure but flips a convention (rotation direction) or
  silently skips one operation; output stays "close" to ground truth
  (same characters, similar structure) but still scores 0.
- **Argument-scope / bracket-matching collapse** (Example B) — as nesting
  gets deep enough (6+), the model starts misattributing arguments to the
  wrong function call and/or dropping entire sub-branches of the
  composition tree, producing outputs that bear little resemblance to the
  ground truth.

Both patterns are visible well within the 4096-token budget — Example A runs
~1200 tokens total, and both Example B samples run ~1200-1850 tokens,
finishing with a clean JSON answer far short of the cap — confirming the
aggregate finding above that token budget is not the bottleneck.
