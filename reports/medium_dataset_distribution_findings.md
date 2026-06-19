# Medium Dataset Distribution Analysis

**Date:** 2026-06-19  
**Context:** Investigating accuracy dips at steps ~1100 and ~1500 observed in Qwen3-1.7B training runs (e.g., `onpolicy_pos_neg_math`, job 8618667).

---

## Dataset Sizes and Step Mapping

- Easy dataset: 10,560 samples → steps 0–659 (batch size 16, `shuffle=False`)
- Medium dataset: 22,252 samples → steps 660–2,049
- Formula: `medium_row = (step - 660) * 16`

| Dip step | Medium row index |
|----------|-----------------|
| 1100     | 7,040           |
| 1500     | 13,440          |

---

## Major Structural Break at Step ~910 (Row ~4,000)

The medium dataset is composed of two distinct source segments, visible via the `solution` field length:

| Segment | Rows | Steps | Avg solution length | Likely source |
|---------|------|-------|---------------------|---------------|
| 1 | 0–3,750 | 660–894 | 142–1,682 chars | MATH-style (worked solutions included) |
| 2 | 4,000+ | 910–2,049 | ~0 chars | Competition-style (answer only) |

This is the largest distribution shift in the dataset. The `extra_info` only contains `{'index': ..., 'split': 'remaining'}` — no explicit source tag — but the solution field availability is a reliable proxy.

---

## Dip at Step ~1100 (Rows 6,500–7,500)

After a region of expression-heavy answers (rows 5,000–6,500, ~47–50% non-integer), there is a local cluster with:

- **non-integer answers drop**: 47–50% → 41–44%
- **medium integer answers rise**: 10–12% → 13–15%
- Problem style: Russian/Chinese olympiad competition problems with large or complex numerical answers

Sample problems in this region (json indices ~7,837–8,487):
- `ans="32"` — digit replacement puzzle (Vasya problem)
- `ans="461538"` — 6-digit number manipulation
- `ans="2592"` — digit replacement in 2016 (Chinese competition)
- `ans="\sqrt{3}"` — circumscribed hexagon geometry

---

## Dip at Step ~1500 (Rows 13,000–14,000)

A larger shift follows a region of notably high expression-answer density (rows 9,000–12,750, **52–60% non-integer**):

- **non-integer answers drop**: 57–60% → 40%  (~17–20pp swing — larger than the step-1100 dip)
- **medium integer answers rise**: 7–9% → 11–17%
- Problem style: Chinese school math and applied arithmetic (decimal answers like 13.2, 51.36)

Sample problems in this region (json indices ~15,340–16,181):
- `ans="13.2"` — quadrilateral geometry (decimal)
- `ans="51.36"` — fruit candy arithmetic (applied Chinese school math)
- `ans="60"` — rhombus area
- `ans="126"` — arithmetic sequence

---

## Interpretation

Both dips align with **sudden increases in numerical (integer/decimal) answer problems** immediately after regions where expression-type answers dominate. Since `shuffle=False`, source-level clustering in the original index order is preserved.

Possible mechanisms:
1. The model adapts its output format to expression-type answers in the preceding region, then loses accuracy when the distribution shifts back to integer/decimal.
2. The problem clusters at these positions are intrinsically harder (e.g., large numerical answers require exact multi-step computation).

The step-1500 dip is expected to be deeper than the step-1100 dip due to the larger magnitude of the distribution shift (~20pp vs ~8pp in non-integer%).

---

## Answer Type Distribution Summary (500-row rolling window)

The full table is available by running:

```bash
python3 << 'EOF'
import json, numpy as np, pandas as pd, re
from collections import Counter

with open('data/math/math-medium/train.json') as f:
    data = json.load(f)
df = pd.read_parquet('data/math/math-medium/train.parquet')
indices = df['extra_info'].apply(lambda x: x['index']).values

def answer_complexity(ans):
    ans = str(ans).strip()
    if re.match(r'^-?\d+$', ans):
        val = abs(int(ans))
        if val <= 100: return 'small_int'
        elif val <= 1000: return 'medium_int'
        else: return 'large_int'
    return 'non_int'

complexities = [answer_complexity(data[i]['answer']) for i in indices]
window = 500
for start in range(0, len(df)-window, window//2):
    chunk = complexities[start:start+window]
    step = 660 + start//16
    print(f'row {start:5d}  step~{step:4d} | ' + ' '.join(f'{k}:{Counter(chunk)[k]/len(chunk)*100:.0f}%' for k in ['small_int','medium_int','large_int','non_int']))
EOF
```
