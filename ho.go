package ho

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"golang.org/x/exp/constraints"
)

//////
// Exported functionalities.
//////

// DefaultConfig returns a default configuration.
func DefaultConfig() OptimizationConfig {
	return OptimizationConfig{
		Iterations:      50,
		InitialSamples:  10,
		NumCandidates:   50,
		AcquisitionFunc: UCB,
		AcqParams: AcquisitionParams{
			BestSoFar:   math.MaxFloat64,
			Beta:        2.0,
			RandomState: rand.New(rand.NewSource(time.Now().UnixNano())),
			Xi:          0.01,
		},
		ProgressChan: nil, // Default to no progress updates.
	}
}

// OptimizeHyperparameters uses Bayesian optimization to find the optimal hyperparameters
// for your benchmark function. It combines Gaussian Process regression with acquisition
// functions to efficiently search the parameter space.
//
// Type Parameter:
//   - T: The numeric type for parameters (int64 or float64)
//
// Parameters:
// - config: OptimizationConfig controlling the optimization process
// - benchmarkFunc: The function whose parameters you want to optimize
// - hypers: One or more ParameterRange defining the search space
//
// Returns:
// - []T: The best parameters found (in same order as hypers)
//
// Usage example:
//
//	// Integer optimization example
//	ranges := []ParameterRange[int64]{
//	    {Min: 1024, Max: 1048576},  // Buffer size (1KB to 1MB)
//	    {Min: 1, Max: 32},          // Worker count
//	}
//
//	intBenchmark := BenchmarkFunc[int64](func(params ...int64) error {
//	    bufferSize := params[0]
//	    workerCount := params[1]
//	    return runWorkload(bufferSize, workerCount)
//	})
//
//	bestIntParams := OptimizeHyperparameters(
//	    DefaultConfig(),
//	    intBenchmark,
//	    ranges...,
//	)
//
//	// Float optimization example
//	floatRanges := []ParameterRange[float64]{
//	    {Min: 0.0001, Max: 0.1},  // Learning rate
//	    {Min: 0.0, Max: 1.0},     // Momentum
//	}
//
//	floatBenchmark := BenchmarkFunc[float64](func(params ...float64) error {
//	    learningRate := params[0]
//	    momentum := params[1]
//	    return trainModel(learningRate, momentum)
//	})
//
//	bestFloatParams := OptimizeHyperparameters(
//	    DefaultConfig(),
//	    floatBenchmark,
//	    floatRanges...,
//	)
//
// How it works:
// 1. Takes InitialSamples random samples to build initial model
// 2. For each iteration:
//   - Generates NumCandidates random candidate points
//   - Uses Gaussian Process to predict performance at each point
//   - Uses AcquisitionFunc to select most promising point
//   - Evaluates the selected point
//   - Updates the model with the new result
//
// 3. Returns the best parameters found
//
// Important notes:
// - Thread-safe: Can be called concurrently with different configs
// - Progressive: Quality typically improves with more iterations
// - Adaptive: Learns from previous evaluations
// - Robust: Handles noisy measurements
//
// Best practices:
// - Start with DefaultConfig() and adjust as needed
// - Use reasonable parameter ranges (too wide = slower convergence)
// - Ensure benchmark function is representative of real workload
// - Consider running multiple optimizations and comparing results
//
// Performance considerations:
// - Total runtime = InitialSamples + Iterations evaluations
// - Each iteration evaluates exactly one point
// - Memory usage scales with number of evaluations
// - Consider reducing NumCandidates if iterations are too slow
func OptimizeHyperparameters[T constraints.Integer | constraints.Float](
	config OptimizationConfig,
	benchmarkFunc BenchmarkFunc[T],
	hypers ...ParameterRange[T],
) []T {
	// Initialize thread-safe random number generator for generating parameter
	// values. Using current time as seed ensures different random sequences
	// across runs.
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	var rngMu sync.Mutex

	// safeRandomParams generates a set of random parameters within the specified ranges
	// in a thread-safe manner. This is used both for initial sampling and generating
	// candidates during optimization.
	//
	// Parameters:
	// - hypers: Slice of ParameterRange defining valid ranges for each parameter
	//
	// Returns:
	// - []T: Slice of random values, one for each parameter range
	safeRandomParams := func(hypers []ParameterRange[T]) []T {
		rngMu.Lock()
		defer rngMu.Unlock()

		params := make([]T, len(hypers))
		for i, hyper := range hypers {
			switch any(hyper.Min).(type) {
			case int, int32, int64:
				// For integer types, generate random integer in range
				min := int64(hyper.Min)
				max := int64(hyper.Max)
				params[i] = T(min + rng.Int63n(max-min+1))
			case float32, float64:
				// For float types, generate random float in range
				min := float64(hyper.Min)
				max := float64(hyper.Max)
				params[i] = T(min + rng.Float64()*(max-min))
			}
		}
		return params
	}

	// Helper function to convert parameters to float64 for Gaussian Process
	paramsToFloat64s := func(params []T) []float64 {
		floats := make([]float64, len(params))
		for i, v := range params {
			floats[i] = float64(v)
		}
		return floats
	}

	// Initialize the Gaussian Process model that will be used to predict
	// performance at untested points.
	gp := newGaussianProcess()

	// bestParams tracks the parameter combination that produced the best result.
	bestParams := make([]T, len(hypers))

	// bestTime tracks the best execution time seen so far (lower is better).
	bestTime := math.MaxFloat64

	// bestMu protects access to bestParams and bestTime.
	var bestMu sync.Mutex

	// Helper function to send progress updates.
	sendProgress := func(phase string, iteration, total int, currentParams []T, execTime float64) {
		if config.ProgressChan != nil {
			bestMu.Lock()

			// Convert current and best params to []int for backward compatibility
			currentInts := make([]int, len(currentParams))
			bestInts := make([]int, len(bestParams))
			for i, v := range currentParams {
				currentInts[i] = int(v)
			}
			for i, v := range bestParams {
				bestInts[i] = int(v)
			}

			update := ProgressUpdate{
				Phase:             phase,
				CurrentIteration:  iteration,
				TotalIterations:   total,
				CurrentParams:     currentInts,
				CurrentBestParams: bestInts,
				CurrentBestTime:   bestTime,
				LastExecutionTime: execTime,
			}

			bestMu.Unlock()

			select {
			case config.ProgressChan <- update:
			default:
				// Skip update if channel is full.
			}
		}
	}

	// updateBest safely updates the best parameters and time if a new best is
	// found.
	//
	// Parameters:
	// - params: Parameter combination to potentially update as best
	// - executionTime: Execution time achieved with these parameters
	updateBest := func(params []T, executionTime float64) {
		bestMu.Lock()
		defer bestMu.Unlock()

		if executionTime < bestTime {
			bestTime = executionTime
			copy(bestParams, params)
		}
	}

	// Phase 1: Initial random sampling.
	//
	// Build initial model by sampling random points in the parameter space.
	// This helps establish a baseline understanding of the function behavior.
	for i := 0; i < config.InitialSamples; i++ {
		// Generate and evaluate random parameters.
		params := safeRandomParams(hypers)

		// Measure execution time directly with our generic benchmark function
		startTime := time.Now()
		err := benchmarkFunc(params...)
		executionTime := float64(time.Since(startTime).Nanoseconds())

		// Apply penalty if the benchmark failed
		if err != nil {
			executionTime = math.MaxFloat64/2 + executionTime
		}

		// Convert parameters to float64 for the Gaussian Process
		floatParams := paramsToFloat64s(params)

		// Update our model with the new observation
		gp.Update(floatParams, executionTime)

		// Update best parameters if this is better
		updateBest(params, executionTime)

		sendProgress("InitialSampling", i+1, config.InitialSamples, params, executionTime)
	}

	// Phase 2: Bayesian optimization loop.
	//
	// Iteratively select and evaluate new points based on model predictions.
	for i := 0; i < config.Iterations; i++ {
		var nextParams []T
		bestAcquisition := math.MaxFloat64

		// Update acquisition function with current best time
		config.AcqParams.BestSoFar = bestTime

		// Generate and evaluate random candidates
		// Choose the most promising one according to the acquisition function
		for j := 0; j < config.NumCandidates; j++ {
			// Generate random candidate parameters
			candidateParams := safeRandomParams(hypers)
			floatCandidateParams := paramsToFloat64s(candidateParams)

			// Get model's prediction for these parameters
			mean, variance := gp.Predict(floatCandidateParams)

			// Evaluate how promising this point is
			acquisition := config.AcquisitionFunc(mean, variance, config.AcqParams)

			// Update if this is the most promising candidate so far
			if acquisition < bestAcquisition {
				bestAcquisition = acquisition
				nextParams = candidateParams
			}
		}

		// Evaluate the most promising candidate
		startTime := time.Now()
		err := benchmarkFunc(nextParams...)
		executionTime := float64(time.Since(startTime).Nanoseconds())

		// Apply penalty if the benchmark failed
		if err != nil {
			executionTime = math.MaxFloat64/2 + executionTime
		}

		// Update model with the new observation
		floatNextParams := paramsToFloat64s(nextParams)
		gp.Update(floatNextParams, executionTime)

		// Update best parameters if this is better
		updateBest(nextParams, executionTime)

		sendProgress("Optimization", i+1, config.Iterations, nextParams, executionTime)
	}

	return bestParams
}
