## Why

`self_consistency_correct` 用 `Counter(str(parsed[-1]))` 对答案做众数投票，但 `parsed[-1]` 是未经归一化的 LaTeX 字符串。数学上等价的答案（如 `1/2`、`2/4`、`0.5`、`\frac{1}{2}`）会被分到不同的 Counter bucket，导致选票分流、众数随机。而 `math_verify.verify()` 之所以能正确判断等价，是因为它使用 `parsed[0]`（sympy 对象），sympy 在 parse 阶段已自动将 `Rational(2,4)` 约分为 `Half`。修复：改为使用 sympy 对象的 `str()` 做 Counter key。

## What Changes

- 修改 `_normalize_parsed`：从 `str(parsed[-1])` 改为 `str(parsed[0])`，对浮点型加 `nsimplify` 强制转有理数

## Capabilities

### New Capabilities

- `answer-normalization`: `_normalize_parsed` 和 `self_consistency_correct` 的答案归一化使用 sympy 对象而非 LaTeX 字符串，确保数学等价答案归入同一 Counter bucket

### Modified Capabilities
<!-- No existing spec covers answer verification -->

## Impact

- `utils/answer_verifier.py` — `_normalize_parsed` 函数（~3 行改动）
- `self_consistency_correct` 的投票精度提升，等价表达式不再分流
- 不影响 `extract_answer` 的行为（返回内容语义不变，只是表示更规范）
