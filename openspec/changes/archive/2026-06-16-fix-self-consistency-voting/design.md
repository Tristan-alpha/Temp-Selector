## Context

`math_verify.parse("$expr$")` 返回 `[sympy_object, latex_string]`。sympy 在构造时自动归一化：

| 输入 | `r[0]` (sympy) | `str(r[0])` | `r[-1]` (LaTeX) | `str(r[-1])` |
|---|---|---|---|---|
| `1/2` | `Half` | `1/2` | `\frac{1}{2}` | `\frac{1}{2}` |
| `2/4` | `Half` | `1/2` | `\frac{2}{4}` | `\frac{2}{4}` |
| `0.5` | `Half` | `1/2` | `\frac{1}{2}` | `\frac{1}{2}` |
| `0.50` | `Float(0.5)` | `0.500...` | `0.50` | `0.50` |

当前 `_normalize_parsed` 用 `str(parsed[-1])`（右列），导致等价表达式分到不同 Counter bucket。`verify()` 内部用 `sympy_str_eq(str(r[0]), str(r[0]))`（中列）做第一层快匹配，已经依靠 sympy 归一化。

## Goals / Non-Goals

**Goals:**
- `Counter` 按数学等价而非文本相等分组
- `self_consistency_correct` 的众数投票不受答案书写形式影响

**Non-Goals:**
- 不修改 `verify()` 或 `parse()` 的行为
- 不改 `extract_answer` 的外部语义

## Decisions

### Decision 1: `parsed[-1]` → `parsed[0]`

使用 sympy 对象的 `str()` 作为规范形式。sympy 在构造 `Rational` 时自动约分（`Rational(2,4)` → `Half`），大部分等价表达式天然归一。

### Decision 2: 对 `Float` 加 `nsimplify`

`0.50` 被 parse 为 `Float` 而非 `Half`，`str(Float)` = `"0.500000000000000"`。通过 `sympy.nsimplify(r[0], [sympy.Rational])` 强制转为有理数。

替代方案：用 `verify()` 做 O(N²) 等价类分组。更精确但更慢，且 `self_consistency_correct` 本身就需要和 gold 比对，不需要"不依赖 gold 的投票"这一属性。

## Risks / Trade-offs

- `nsimplify` 可能把某些表达式转成非预期形式。仅在 `isinstance(r[0], Float)` 时调用，范围受限
- 变量名（如 `x` vs `y`）在 sympy `str()` 中保持原样，不会错误合并——这是期望行为
