# ho — critical defects found in review (base 2f526cf8)

Scope: `github.com/thalesfsp/ho` at base `2f526cf89e891550a1b308f532aa29009f1617d8`.
Only defects with a reproducible trigger and severe impact are listed. Both defects are now
fixed locally on `fix/critical-findings`, pending independent review and release.
The original base-referenced diagnosis and plan are preserved below.
Reproducers and raw logs live in `/Volumes/Dev/Dev/critical-review-20261116/ho/`
(`probes/consumer/main.go`, `probes/pkgcopy/mechanism_test.go`, `logs/probe-*.log`).

---

## C1 — `OptimizeHyperparameters` panics with `DefaultConfig()` on small, valid ranges

**Severity: CRITICAL.** The only public entry point crashes the caller's process. It takes
`DefaultConfig()` and a valid range (`Min <= Max`). The function has no error return, so
callers cannot guard against it short of `recover`. Reproduced 20/20 runs in each of 4 triggers.
The wide-range control failed 0/20.

**Source.**
- `gaussianProcess.go:172-179`: `variance = 1 - (sum k)^2 / n`. This goes negative once a
  candidate sits close to more than sqrt(n) observations.
- `acquisitionfunction.go:35, 65, 98, 135`: every built-in acquisition takes `math.Sqrt(variance)`,
  which gives NaN for negative variance.
- `gaussianProcess.go:164-170` with `ho.go:256` and `ho.go:313`: a failed benchmark is
  scored `math.MaxFloat64/2`. Summing sufficiently many weighted terms (more than two at weight 1) in `sum += k[i]*gp.Y[i]`
  overflows to `+Inf`, so the acquisition becomes `+Inf` or NaN.
- `ho.go:275-302`: `nextParams` starts as `nil`. It is assigned only when
  `acquisition < bestAcquisition`, and `bestAcquisition` starts at `math.MaxFloat64`. NaN and `+Inf`
  never pass that check, so when no candidate is finite `nextParams` stays `nil`.
- `ho.go:307`: `benchmarkFunc(nextParams...)` is then called with **zero** params. A benchmark
  written like the documented examples (`params[0]`) panics with `index out of range [0] with length 0`.
- `ho.go:317-319`: if the benchmark tolerates zero params, `gp.Update` stores a 0-length
  vector. The next `Predict` then panics at `gaussianProcess.go:88-90` with
  `input vectors must have the same length`.
- The same nil path also fires when `NumCandidates <= 0`.

**Triggers (all reproduced with `DefaultConfig()`, 20/20 runs each):**
1. `ParameterRange[int]{1,3}`, `{1,3}` (two dimensions): user-code index panic.
2. Same ranges, with a benchmark that ignores its params: library panic in `RBFKernel`.
3. `ParameterRange[int]{1,5}` (one dimension).
4. `ParameterRange[int]{1,20}` with a benchmark that always returns an error (penalty overflow).
5. Any C2 type (all samples identical gives variance `1-n`).

**Evidence.** `logs/probe-base.log` and `logs/probe-after-*.log` (unchanged after the dependency update).
`logs/probe-mechanism.log` shows, for the 3x3 grid: variance between -0.16 and -2.48 at every
candidate, `UCB = PI = EI = TS = NaN` at 9/9 candidates, and penalty mean `+Inf`.

**Root cause.**
1. The GP variance formula is not bounded below.
2. The penalty value is large enough to overflow when summed.
3. The selection loop has no fallback for non-finite scores, and no guard for `NumCandidates < 1`.

**Bounded fix plan** (`gaussianProcess.go`, `ho.go`, and acquisition zero-variance guards in
`acquisitionfunction.go`; preserve the public API):
1. `Predict`: clamp `variance` to `[0, 1]`, so it is never negative. Define zero-variance
   limits in PI/EI before dividing by the standard deviation; a clamp alone leaves `0/0`
   possible when improvement is exactly zero. Assert finite outputs for that case.
2. `Predict`: compute the mean overflow-safely, either as `sum += (k[i]/n) * gp.Y[i]` or with
   the penalty replaced (step 3).
3. Keep failure ranking explicit while making aggregation overflow-safe. Do not assume a fixed
   smaller penalty such as `1e18` exceeds every successful duration: `time.Duration` permits
   larger values. Verify a failure always ranks worse than every successful observation and
   test the all-failures path alongside step 2.
4. Selection loop: reject non-finite acquisition scores. Fall back to the first generated candidate
   when no candidate wins (`nextParams == nil`). Treat `NumCandidates < 1` as 1.

**Regression and negative-control plan.**
- External `ho_test` table test over triggers 1–4 above, with a benchmark that indexes `params`.
  It asserts: no panic, `len(result) == len(hypers)`, and every returned value and every value
  passed to the benchmark lies in `[Min, Max]`.
- In-package test with the 3x3-grid fixture from `probes/pkgcopy/mechanism_test.go`. It asserts
  `Predict` variance is in `[0,1]` and every built-in acquisition is finite.
- Include PI/EI with zero variance and exactly zero improvement, plus at least three identical
  failure observations to exercise overflow.
- **Negative control:** revert each relevant guard independently, and confirm the tests fail with the
  panics recorded in `logs/probe-base.log`.
- **Bypass test:** the all-candidates-`+Inf` path (trigger 4, an always-failing benchmark),
  plus `NumCandidates = 0`. Together they prove the fallback handles `+Inf` and empty candidate
  sets, not only NaN.

---

## C2 — integer and float types outside a hard-coded type switch always sample `0`

**Severity: CRITICAL (data integrity).** The exported constraint is
`constraints.Integer | constraints.Float` (`ho.go:114`, `types.go:64`, `types.go:112`). It admits
`int8`, `int16`, `uint`, `uint8` through `uint64`, `uintptr`, and **every named type** such as
`type Workers int`. For all of these, every sampled parameter is the zero value, even when the
declared inclusive range excludes it. So:
- the benchmark is never run with an in-range value;
- with `DefaultConfig()` the run then panics through C1;
- with `Iterations: 0` it silently returns an out-of-range "best", for example `[0]` for
  `{Min: 4, Max: 8}`.

**Source.** `ho.go:142-157`: `switch any(hyper.Min).(type) { case int, int32, int64: … case float32, float64: … }`.
It has no default case, so `params[i]` keeps `T`'s zero value. Named types never match a concrete
case: their dynamic type is the named type itself.

**Triggers (20/20 runs each, int control 0/20):**
- `type Workers int`, `ParameterRange[Workers]{4, 8}`:
  - `DefaultConfig()`: the benchmark sees only `0` (`seen=map[0:10]`), then panics (C1).
  - `Iterations = 0`: returns `[0]`.
- `ParameterRange[uint]{10, 20}` with `Iterations = 0`: returns `[0]`, and the benchmark only ever saw `0`.

**Evidence.** `logs/probe-base.log` and `logs/probe-after-C2-*.log`.

**Root cause.** The code checks the runtime concrete type, but the generic constraint admits a
whole type set. The silent no-match path has no default case.

**Bounded fix plan** (only `ho.go` `safeRandomParams`):
- Decide integer vs float generically. Either use `reflect.ValueOf(hyper.Min).Kind()` with all
  `Int*`, `Uint*` and `Float*` kinds, or use the type-set-safe probe `T(1)/T(2) == 0`, which is
  true for integers.
- Integers: compute the span in `uint64` arithmetic. If `span+1` overflows or exceeds `MaxInt63`,
  use `rng.Uint64()` with rejection sampling. Otherwise use `Int63n(span+1)`.
- Floats: keep the existing formula.
- Add a defensive `default` that panics with a clear message instead of returning zeros silently.

**Regression and negative-control plan.**
- Table test over `int8`, `int16`, `uint`, `uint8`, `uint16`, `uint32`, `uint64`,
  `type Workers int` and `type Rate float64`, with `Iterations: 0` and with `DefaultConfig()`
  (after C1). It asserts every value passed to the benchmark and every returned value lies in
  `[Min, Max]`, and that a span of 1000 or more yields more than one distinct value over 10 samples.
- **Negative control:** restore the old switch and confirm the test fails with zeros.
- **Bypass tests:**
  - `ParameterRange[uint64]{MaxUint64-10, MaxUint64}` must stay in range with no overflow.
  - `ParameterRange[int64]{MinInt64, MaxInt64}` must not panic. Today `ho.go:149` computes
    `max-min+1 == 0`, which makes `Int63n(0)` panic.


## Implementation and regression evidence

- **C1 implemented:** scale GP mean terms before summation, clamp negative variance,
  define PI/EI zero-variance limits, normalize nonpositive candidate counts to one,
  and retain an in-range fallback when all scores are nonfinite. Failure penalties
  remain above all successful `time.Duration` measurements; no scoring constant changed.
- **C2 implemented:** distinguish the numeric type sets by integer versus floating
  division; preserve named types. Use unsigned modular arithmetic plus rejection
  sampling for all integer widths, including the full signed/unsigned 64-bit range.
  Interpolate floats without overflowing the range width and clamp rounding at bounds.
- Strengthen all four original tests with parameter-bound assertions; add focused
  regressions for singleton/default ranges across all acquisitions, failed benchmarks,
  nonfinite scores, candidate counts, supported numeric types and extreme/equal bounds.
- Negative controls restore original source (three repeated runs) and independently
  remove mean scaling, variance clamping, candidate fallback/normalization and
  acquisition zero-variance guards. All fail for the expected reason. Raw evidence:
  `/Volumes/Dev/Dev/critical-review-20261116/ho/fix-negative-controls.json`.
- Keep all 38 configured lint checks. Fix the one inherited whitespace issue;
  `make ci` now includes unit tests and the workflow runs race tests.
- Commit/CI/review checkpoint and final raw logs are recorded outside the repo in
  `/Volumes/Dev/Dev/critical-review-20261116/ho/fix-report.json`.
