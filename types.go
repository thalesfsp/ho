package ho

import (
	"math/rand"

	"golang.org/x/exp/constraints"
)

// ProgressUpdate represents the current state of the optimization process.
type ProgressUpdate struct {
	// Phase indicates whether we're in initial sampling or optimization phase
	Phase string

	// CurrentIteration is the current iteration number
	CurrentIteration int

	// TotalIterations is the total number of iterations to run
	TotalIterations int

	// CurrentParams holds the parameter values being tested
	CurrentParams []int

	// CurrentBestParams holds the best parameters found so far
	CurrentBestParams []int

	// CurrentBestTime holds the best execution time found so far
	CurrentBestTime float64

	// LastExecutionTime holds the execution time of the last test
	LastExecutionTime float64
}

// ParameterRange defines the valid range for a hyperparameter in the optimization process.
// Each hyperparameter must have a minimum and maximum value to define its search space.
//
// Type Parameter:
//   - T: The numeric type for this parameter range (int64 or float64)
//
// Fields:
// - Min: The minimum (inclusive) value for this hyperparameter
// - Max: The maximum (inclusive) value for this hyperparameter
//
// Usage:
//
//	// Example 1: Buffer size range from 1KB to 1MB
//	bufferSizeRange := ParameterRange[int64]{
//	    Min: 1024,      // 1KB
//	    Max: 1048576,   // 1MB
//	}
//
//	// Example 2: Learning rate range from 0.0001 to 0.1
//	learningRateRange := ParameterRange[float64]{
//	    Min: 0.0001,
//	    Max: 0.1,
//	}
//
// Validation:
// - Min must be less than or equal to Max
// - The range is inclusive of both Min and Max values
//
// Warning:
//   - Using a very large range may result in slower convergence
//     as the search space becomes too large to explore effectively
type ParameterRange[T constraints.Integer | constraints.Float] struct {
	// Min defines the minimum allowed value (inclusive) for this hyperparameter.
	// Example: Min: 1 means the hyperparameter cannot be less than 1
	Min T

	// Max defines the maximum allowed value (inclusive) for this hyperparameter.
	// Example: Max: 100 means the hyperparameter cannot exceed 100
	Max T
}

// BenchmarkFunc defines the signature for functions that will be optimized.
// This function type represents the task whose parameters you want to optimize.
//
// Type Parameter:
//   - T: The numeric type for parameters (int64 or float64)
//
// Parameters:
//   - params: Variable number of numeric parameters representing the hyperparameters
//     to be optimized. The number of parameters must match the number of
//     ParameterRange values provided to OptimizeHyperparameters.
//
// Returns:
// - error: Return nil if the benchmark succeeded, or an error if it failed
//
// Usage example:
//
//	// Example 1: Optimizing buffer size and worker count with integers
//	intBenchmark := BenchmarkFunc[int64](func(params ...int64) error {
//	    bufferSize := params[0]    // First parameter
//	    workerCount := params[1]   // Second parameter
//
//	    // Your actual benchmark code here
//	    result, err := runYourWorkload(bufferSize, workerCount)
//	    if err != nil {
//	        return fmt.Errorf("benchmark failed: %w", err)
//	    }
//
//	    return nil
//	})
//
//	// Example 2: Optimizing learning parameters with float64
//	floatBenchmark := BenchmarkFunc[float64](func(params ...float64) error {
//	    learningRate := params[0]
//	    momentum := params[1]
//
//	    // ... test model performance ...
//	    return nil
//	})
type BenchmarkFunc[T constraints.Integer | constraints.Float] func(params ...T) error

// AcquisitionFunc defines the signature for acquisition functions used in the
// Bayesian optimization process. These functions help decide which points in the
// parameter space should be evaluated next.
//
// Parameters:
// - mean: The predicted mean performance at a point (lower is better)
// - variance: The predicted variance/uncertainty at that point
// - params: Additional parameters needed by specific acquisition functions
//
// Returns:
// - float64: Acquisition value (lower values indicate more promising points)
//
// Built-in acquisition functions:
// - UCB: Upper Confidence Bound
// - ProbabilityOfImprovement: Probability of finding better value
// - ExpectedImprovement: Expected magnitude of improvement
// - ThompsonSampling: Random sampling from posterior
//
// Usage example:
//
//	// Example 1: Using a built-in acquisition function
//	config := OptimizationConfig{
//	    AcquisitionFunc: UCB,
//	    AcqParams: AcquisitionParams{
//	        Beta: 2.0,
//	    },
//	}
//
//	// Example 2: Custom acquisition function
//	custom := func(mean, variance float64, params AcquisitionParams) float64 {
//	    // Your custom acquisition logic here
//	    return mean - params.Beta * math.Sqrt(variance)
//	}
//	config.AcquisitionFunc = custom
//
// Implementation notes for custom acquisition functions:
// - Should handle edge cases (zero variance, extreme means)
// - Must be thread-safe
// - Should be deterministic
// - Should return lower values for more promising points
// - Must properly use parameters from AcquisitionParams.
type AcquisitionFunc func(mean, variance float64, params AcquisitionParams) float64

// AcquisitionParams holds parameters used by different acquisition functions to make decisions
// about which points to sample next in the optimization process. Each acquisition function
// may use different parameters to balance between exploring new areas (exploration) and
// focusing on areas known to be good (exploitation).
type AcquisitionParams struct {
	// Beta controls the exploration-exploitation trade-off in the Upper Confidence Bound (UCB)
	// acquisition function.
	// - Higher values (e.g., 3.0 or 5.0) encourage more exploration of uncertain areas
	// - Lower values (e.g., 0.1 or 0.5) focus more on exploiting known good areas
	// Typical values range from 0.1 to 5.0, with 2.0 being a good default.
	Beta float64

	// Xi (Greek letter ξ) is an exploration parameter used in Probability of Improvement (PI)
	// and Expected Improvement (EI) acquisition functions. It controls how much improvement
	// we want over the current best observation.
	// - Higher values (e.g., 0.1) encourage more exploration
	// - Lower values (e.g., 0.01) focus more on local optimization
	// Typical values range from 0.01 to 0.1.
	Xi float64

	// BestSoFar keeps track of the best (lowest) execution time we've seen so far.
	// This is used by PI and EI to determine how much improvement a new point might offer.
	//
	// Initial value:
	// - MUST be set to math.MaxFloat64 in your configuration
	// - Will be automatically updated by the optimizer during the optimization process
	// - The units are in nanoseconds (same as time.Duration)
	//
	// Example:
	//     params := AcquisitionParams{
	//         BestSoFar: math.MaxFloat64,  // Required initial value
	//     }
	BestSoFar float64

	// RandomState is the random number generator used by Thompson Sampling.
	//
	// Required initialization:
	// - MUST be initialized using rand.New(rand.NewSource(seed))
	// - Typically use time.Now().UnixNano() as the seed
	// - Each optimization run should have its own RandomState
	//
	// Example:
	//     params := AcquisitionParams{
	//         RandomState: rand.New(rand.NewSource(time.Now().UnixNano())),
	//     }
	//
	// Warning:
	// - Do NOT use a nil RandomState
	// - Do NOT share RandomState between different optimization runs
	RandomState *rand.Rand
}

// OptimizationConfig holds all configuration parameters for the Bayesian optimization process.
// It allows you to control how the optimization behaves, including its thoroughness,
// exploration strategy, and computational budget.
//
// Fields explanation:
// - Iterations: Number of optimization steps after initial sampling
// - InitialSamples: Number of random samples to take before starting optimization
// - NumCandidates: Number of random candidates to evaluate per iteration
// - AcquisitionFunc: Strategy for choosing next points to evaluate
// - AcqParams: Parameters for the acquisition function
//
// Usage example:
//
//	config := OptimizationConfig{
//	    // Run optimization for 50 iterations
//	    Iterations: 50,
//
//	    // Start with 10 random samples to build initial model
//	    InitialSamples: 10,
//
//	    // Consider 100 random candidates per iteration
//	    NumCandidates: 100,
//
//	    // Use Expected Improvement strategy
//	    AcquisitionFunc: ExpectedImprovement,
//
//	    // Configure acquisition function parameters
//	    AcqParams: AcquisitionParams{
//	        Xi: 0.01,
//	        BestSoFar: math.MaxFloat64,
//	        RandomState: rand.New(rand.NewSource(time.Now().UnixNano())),
//	    },
//	}
//
// Default values recommendations:
// - Iterations: 50 (increase for more thorough optimization)
// - InitialSamples: 10 (increase for more stable initial model)
// - NumCandidates: 50-100 (increase for more thorough search per iteration)
//
// Performance impact notes:
// - Higher Iterations = Better results but longer total runtime
// - Higher InitialSamples = Better model but more initial overhead
// - Higher NumCandidates = Better per-iteration results but slower iterations
//
// Note:
// - Create separate configs for parallel optimizations.
type OptimizationConfig struct {
	// Iterations determines how many optimization steps to perform after the
	// initial sampling phase. Each iteration involves evaluating NumCandidates
	// points and selecting the best one to actually benchmark.
	// Recommended range: 20-200
	Iterations int

	// InitialSamples determines how many random points to evaluate before
	// starting the optimization process. These samples help build the initial
	// Gaussian Process model.
	// Recommended range: 5-20
	InitialSamples int

	// NumCandidates determines how many random candidates to consider in each
	// iteration before selecting the best one to evaluate.
	// Higher values = more thorough search but slower iterations.
	// Recommended range: 50-500
	NumCandidates int

	// AcquisitionFunc determines the strategy for selecting the next point to
	// evaluate. See AcquisitionFunc type for built-in options.
	AcquisitionFunc AcquisitionFunc

	// AcqParams holds the parameters for the acquisition function.
	// Must be properly initialized based on the chosen AcquisitionFunc.
	AcqParams AcquisitionParams

	// ProgressChan is used to send progress updates during optimization
	// If nil, no updates will be sent
	ProgressChan chan<- ProgressUpdate
}
