package ho

import (
	"errors"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/constraints"
)

func TestOptimizeNarrowRanges(t *testing.T) {
	for name, acquisition := range map[string]AcquisitionFunc{
		"ucb": UCB, "pi": ProbabilityOfImprovement,
		"ei": ExpectedImprovement, "thompson": ThompsonSampling,
	} {
		t.Run(name, func(t *testing.T) {
			for _, failing := range []bool{false, true} {
				cfg := DefaultConfig()
				cfg.AcquisitionFunc = acquisition
				calls := 0
				assert.NotPanics(t, func() {
					best := OptimizeHyperparameters(cfg, func(ps ...int) error {
						require.Equal(t, []int{4, 6}, ps)
						calls++

						if failing {
							return errors.New("benchmark failed")
						}

						return nil
					}, ParameterRange[int]{Min: 4, Max: 4}, ParameterRange[int]{Min: 6, Max: 6})
					require.Equal(t, []int{4, 6}, best)
				})
				assert.Equal(t, cfg.InitialSamples+cfg.Iterations, calls)
			}
		})
	}
}

func TestOptimizeCandidateFallback(t *testing.T) {
	for name, score := range map[string]float64{"nan": math.NaN(), "positive_inf": math.Inf(1), "negative_inf": math.Inf(-1)} {
		t.Run(name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.InitialSamples, cfg.Iterations = 0, 3
			cfg.AcquisitionFunc = func(float64, float64, AcquisitionParams) float64 { return score }
			checkSampling(t, cfg, ParameterRange[int]{Min: 4, Max: 4})
		})
	}

	for _, candidates := range []int{0, -1} {
		cfg := DefaultConfig()
		cfg.InitialSamples, cfg.Iterations, cfg.NumCandidates = 0, 3, candidates
		checkSampling(t, cfg, ParameterRange[int]{Min: 4, Max: 4})
	}
}

func TestPredictFinite(t *testing.T) {
	gp := newGaussianProcess()
	for i := 0; i < 10; i++ {
		gp.Update([]float64{4}, math.MaxFloat64/2)
	}

	mean, variance := gp.Predict([]float64{4})
	require.False(t, math.IsNaN(mean) || math.IsInf(mean, 0))
	assert.InDelta(t, 1, mean/(math.MaxFloat64/2), 1e-12)
	assert.GreaterOrEqual(t, variance, float64(0))
	assert.LessOrEqual(t, variance, float64(1))
}

func TestPredictBoundedVariance(t *testing.T) {
	gp := newGaussianProcess()
	for i := 0; i < 10; i++ {
		gp.Update([]float64{4}, 1)
	}

	_, variance := gp.Predict([]float64{4})
	assert.GreaterOrEqual(t, variance, float64(0))
	assert.LessOrEqual(t, variance, float64(1))
}

func TestZeroVarianceAcquisitions(t *testing.T) {
	params := AcquisitionParams{BestSoFar: 1}
	for _, variance := range []float64{0, -1} {
		for _, tc := range []struct{ mean, pi, ei float64 }{{0, 0, 0}, {1, 0.5, 0}, {2, 1, 1}} {
			assert.Equal(t, tc.pi, ProbabilityOfImprovement(tc.mean, variance, params))
			assert.Equal(t, tc.ei, ExpectedImprovement(tc.mean, variance, params))
		}
	}
}

func TestPositiveVarianceAcquisitions(t *testing.T) {
	params := AcquisitionParams{BestSoFar: 1}
	assert.Equal(t, 0.5, ProbabilityOfImprovement(1, 1, params))
	assert.InDelta(t, 1/math.Sqrt(2*math.Pi), ExpectedImprovement(1, 1, params), 1e-12)
}

func TestPredictionKernelWidth(t *testing.T) {
	gp := newGaussianProcess()
	gp.Update([]float64{0}, 8)
	narrowMean, narrowVariance := gp.Predict([]float64{4})
	gp.SetSigma(4)
	assert.Equal(t, float64(4), gp.GetSigma())
	wideMean, wideVariance := gp.Predict([]float64{4})
	assert.Greater(t, wideMean, narrowMean)
	assert.Less(t, wideVariance, narrowVariance)
}

type testWorkers int

type testRate float64

func TestSamplingNumericTypes(t *testing.T) {
	for _, iterations := range []int{0, 3} {
		cfg := DefaultConfig()
		cfg.InitialSamples, cfg.Iterations, cfg.NumCandidates = 6, iterations, 10
		checkSampling(t, cfg, ParameterRange[int8]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[int16]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uint]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uint8]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uint16]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uint32]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uint64]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[uintptr]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[testWorkers]{Min: 4, Max: 8})
		checkSampling(t, cfg, ParameterRange[testRate]{Min: 4, Max: 8})
	}
}

func TestSamplingExtremeRanges(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InitialSamples, cfg.Iterations = 128, 0
	checkSampling(t, cfg, ParameterRange[int64]{Min: math.MinInt64, Max: math.MaxInt64})
	checkSampling(t, cfg, ParameterRange[uint64]{Min: 0, Max: math.MaxUint64})
	checkSampling(t, cfg, ParameterRange[uint64]{Min: 0, Max: 1 << 63})
	checkSampling(t, cfg, ParameterRange[uint64]{Min: math.MaxUint64 - 10, Max: math.MaxUint64})
	checkSampling(t, cfg, ParameterRange[int64]{Min: -5, Max: 5})
	checkSampling(t, cfg, ParameterRange[testRate]{Min: -math.MaxFloat64, Max: math.MaxFloat64})
}

func checkSampling[T constraints.Integer | constraints.Float](t *testing.T, cfg OptimizationConfig, bounds ParameterRange[T]) {
	t.Helper()

	calls := 0
	best := OptimizeHyperparameters(cfg, func(ps ...T) error {
		require.Len(t, ps, 1)
		require.GreaterOrEqual(t, ps[0], bounds.Min)
		require.LessOrEqual(t, ps[0], bounds.Max)
		calls++

		return nil
	}, bounds)
	require.Len(t, best, 1)
	assert.GreaterOrEqual(t, best[0], bounds.Min)
	assert.LessOrEqual(t, best[0], bounds.Max)
	assert.Equal(t, cfg.InitialSamples+cfg.Iterations, calls)
}

func TestOptimizeConcurrentProgress(t *testing.T) {
	finished := make(chan []int, 2)
	for i := 0; i < 2; i++ {
		go func() {
			cfg := DefaultConfig()
			cfg.InitialSamples, cfg.Iterations = 3, 5
			cfg.ProgressChan = make(chan ProgressUpdate)
			cfg.AcqParams.RandomState = rand.New(rand.NewSource(1))
			finished <- OptimizeHyperparameters(cfg, func(ps ...int) error { return nil }, ParameterRange[int]{Min: 4, Max: 4})
		}()
	}

	for i := 0; i < 2; i++ {
		select {
		case best := <-finished:
			require.Equal(t, []int{4}, best)
		case <-time.After(5 * time.Second):
			t.Fatal("optimization blocked with an unread progress channel")
		}
	}
}
