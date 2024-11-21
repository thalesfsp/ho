package ho

import (
	"math"
	"time"
)

//////
// Helper functions.
//////

// Helper function used by PI and EI to compute the cumulative distribution
// function of the standard normal distribution.
//
// Returns:
// - Probability that a standard normal random variable is less than x.
func normalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}

// Helper function used by EI to compute the probability density function
// of the standard normal distribution.
//
// Returns:
// - Value of the standard normal PDF at x.
func normalPDF(x float64) float64 {
	return math.Exp(-x*x/2.0) / math.Sqrt(2.0*math.Pi)
}

// measureExecutionTime runs a benchmark function with the given parameters and
// measures its execution time in nanoseconds.
//
// Parameters:
// - f: The benchmark function to measure (must implement BenchmarkFunc interface)
// - params: Slice of integer parameters to pass to the benchmark function
//
// Returns:
// - float64: Execution time in nanoseconds
// - error: Error from benchmark function if it failed, nil otherwise
//
// Important notes:
// - Time measurement includes only the execution of f, not parameter preparation
// - Returns time as float64 for compatibility with Gaussian Process calculations
// - A return value of 0 always indicates an error occurred
// - Time is measured using time.Now() and time.Since() for high precision
//
// Thread safety:
// - This function is thread-safe if and only if the provided benchmark function is thread-safe
// - The time measurement itself is thread-safe
//
// Best practices:
// - Ensure benchmark function measures representative workload
// - Consider running multiple times and averaging for noisy benchmarks
// - Be aware of system noise affecting measurements
// - For very fast operations, consider running multiple iterations within the benchmark.
func measureExecutionTime(f BenchmarkFunc[int], params []int) float64 {
	// Record start time with high precision
	start := time.Now()

	// Execute the benchmark function with provided parameters
	err := f(params...)

	// Calculate total duration
	duration := time.Since(start)

	if err != nil {
		// Instead of returning 0, return a high penalty value
		// This helps the Gaussian Process learn to avoid failing configurations
		// We use MaxFloat64/2 to leave room for adding the actual duration
		penaltyValue := math.MaxFloat64 / 2

		// Add the actual duration to the penalty
		// This helps differentiate between failures that took different amounts of time
		penaltyValue += float64(duration.Nanoseconds())

		return penaltyValue
	}

	return float64(duration.Nanoseconds())
}

// intsToFloats converts a slice of integers to a slice of float64 values.
// This conversion is necessary because the Gaussian Process model works with
// float64 values, but our hyperparameters are integers.
//
// Parameters:
// - ints: Slice of integers to convert
//
// Returns:
// - []float64: New slice containing float64 versions of input values
//
// Important notes:
// - Creates a new slice; doesn't modify the input
// - Preserves order of elements
// - No precision loss (integers convert exactly to float64)
// - Returns empty slice if input is nil or empty
//
// Performance considerations:
// - Allocates new slice of same length as input
// - O(n) time complexity where n is len(ints)
// - Single pass through the data
//
// Thread safety:
// - Function is inherently thread-safe (no shared state)
// - Input slice is not modified
// - Returns new, independent slice.
func intsToFloats(ints []int) []float64 {
	// Allocate new slice with same length as input
	floats := make([]float64, len(ints))

	// Convert each integer to float64
	// This conversion is exact for all integers that fit in int64
	for i, v := range ints {
		floats[i] = float64(v)
	}

	return floats
}
