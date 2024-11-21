package ho

import (
	"math"
	"sync"
)

//////
// Const, vars, types.
//////

// gaussianProcess implements a thread-safe Gaussian Process model for regression
// with multidimensional inputs. It is used to predict the performance of untested
// hyperparameter combinations based on previously observed results.
//
// Fields:
// - mu: RWMutex for thread-safe access to all fields
// - X: Slice of observed input points (each point is a slice of float64)
// - Y: Slice of observed values (execution times) at each input point
// - sigma: Kernel width parameter controlling the smoothness of interpolation
//
// Thread safety:
// - All fields are protected by the RWMutex
// - Safe for concurrent access from multiple goroutines
// - Uses RLock for read operations (Predict, RBFKernel)
// - Uses Lock for write operations (Update, SetSigma)
//
// Memory usage:
// - Grows linearly with number of observations
// - Each observation stores a copy of input parameters
// - O(n) memory where n is number of observations.
type gaussianProcess struct {
	// mu protects access to all fields
	mu sync.RWMutex

	// X stores the input points (hyperparameter combinations)
	// Each element is a slice of float64 values
	// Length of inner slices must be consistent
	X [][]float64

	// Y stores the observed values (execution times) at each point in X
	// Must have same length as X
	Y []float64

	// sigma is the kernel width parameter
	// Larger values = smoother interpolation
	// Smaller values = more local influence
	sigma float64
}

//////
// Methods.
//////

// RBFKernel implements the Radial Basis Function (also known as Gaussian) kernel.
// This kernel measures the similarity between two points in the input space,
// with the similarity decreasing exponentially with distance.
//
// Parameters:
// - x1, x2: Input vectors to compare (must have same length)
//
// Returns:
// - float64: Kernel value (similarity) between the points (0.0 to 1.0)
//
// Usage example:
//
//	gp := newGaussianProcess()
//	similarity := gp.RBFKernel(
//	    []float64{1.0, 2.0},
//	    []float64{1.1, 2.1},
//	)
//
// Mathematical formula:
//
//	k(x1, x2) = exp(-sum((x1 - x2)^2) / (2 * sigma^2))
//
// Important notes:
// - Panics if input vectors have different lengths
// - Returns 1.0 for identical points
// - Returns values close to 0.0 for distant points
// - Uses read lock to access sigma
//
// Thread safety:
// - Protected by read mutex for sigma access
// - Safe for concurrent access
// - Multiple kernel calculations can proceed in parallel.
func (gp *gaussianProcess) RBFKernel(x1, x2 []float64) float64 {
	if len(x1) != len(x2) {
		panic("input vectors must have the same length")
	}

	// Get sigma value thread-safely
	gp.mu.RLock()
	sigma := gp.sigma
	gp.mu.RUnlock()

	// Calculate squared Euclidean distance
	var sum float64

	for i := range x1 {
		diff := x1[i] - x2[i]

		sum += diff * diff
	}

	// Apply RBF kernel formula
	return math.Exp(-sum / (2 * sigma * sigma))
}

// Predict estimates the expected execution time and uncertainty at a given point
// based on previously observed data points.
//
// Parameters:
// - x: Input point at which to make prediction (hyperparameter combination)
//
// Returns:
// - mean: Expected execution time at the input point
// - variance: Uncertainty in the prediction (higher = less certain)
//
// Usage example:
//
//	gp := newGaussianProcess()
//	// Add some observations...
//	mean, variance := gp.Predict([]float64{1.0, 2.0})
//	fmt.Printf("Expected time: %v ± %v\n", mean, math.Sqrt(variance))
//
// Mathematical details:
// - Uses RBF kernel to measure similarity to known points
// - Mean is weighted average of observed values
// - Variance indicates prediction uncertainty
// - Returns (0, 1) if no observations exist
//
// Important notes:
// - Thread-safe (uses read lock)
// - O(n) space complexity for temporary storage
// - O(n^2) time complexity for variance calculation
// - n is the number of observations
//
// Best practices:
// - Check variance to assess prediction reliability
// - Be cautious of predictions far from observed points
// - Consider uncertainty in optimization decisions
//
// Performance considerations:
// - Computation time increases quadratically with observations
// - Consider limiting total observations in long-running optimizations
// - Memory usage is linear with number of observations.
func (gp *gaussianProcess) Predict(x []float64) (mean, variance float64) {
	gp.mu.RLock()
	defer gp.mu.RUnlock()

	// Handle case with no observations
	if len(gp.X) == 0 {
		return 0, 1
	}

	// Calculate kernel values between x and all observed points
	k := make([]float64, len(gp.X))
	for i := range gp.X {
		k[i] = gp.RBFKernel(x, gp.X[i])
	}

	// Calculate mean prediction
	var sum float64

	for i := range gp.X {
		sum += k[i] * gp.Y[i]
	}

	mean = sum / float64(len(gp.X))

	// Calculate variance.
	variance = 1.0

	for i := range gp.X {
		for j := range gp.X {
			variance -= k[i] * k[j] / float64(len(gp.X))
		}
	}

	return mean, variance
}

// Update adds a new observation point to the Gaussian Process model.
// This method is used to train the model with new data points as they are observed
// during the optimization process.
//
// Parameters:
// - x: Slice of float64 values representing the input point (hyperparameters)
// - y: Observed value (execution time) at point x
//
// Usage example:
//
//	gp := newGaussianProcess()
//
//	// Add observation: parameters [1.0, 2.0] resulted in execution time 100.5
//	gp.Update([]float64{1.0, 2.0}, 100.5)
//
// Important notes:
// - Creates a deep copy of input slice x to prevent external modifications
// - Maintains thread safety using mutex
// - Appends to internal X and Y slices
// - Memory usage grows with each update
//
// Thread safety:
// - Protected by write mutex (gp.mu)
// - Safe for concurrent access from multiple goroutines
// - Blocks other Updates and SetSigma operations while running
// - Blocks Predict operations while running
//
// Performance considerations:
// - O(1) time complexity for the update itself
// - Memory grows linearly with number of observations
// - Creates new slice and copies data on each call
// - Consider memory impact with large numbers of updates.
func (gp *gaussianProcess) Update(x []float64, y float64) {
	gp.mu.Lock()
	defer gp.mu.Unlock()

	// Create deep copy of input to prevent external modifications
	newX := make([]float64, len(x))
	copy(newX, x)

	// Append new observation to our training data
	gp.X = append(gp.X, newX)
	gp.Y = append(gp.Y, y)
}

// SetSigma updates the kernel width parameter (sigma) of the Gaussian Process.
// This parameter controls the smoothness of the resulting model and the extent
// of influence of each observation.
//
// Parameters:
// - sigma: New kernel width value (must be positive)
//
// Usage example:
//
//	gp := newGaussianProcess()
//
//	// Set wider kernel for smoother interpolation
//	gp.SetSigma(2.0)
//
//	// Set narrower kernel for more local influence
//	gp.SetSigma(0.5)
//
// Important notes:
// - Affects all subsequent predictions
// - Larger values = smoother interpolation
// - Smaller values = more local influence
// - No validation of sigma value (caller's responsibility)
//
// Thread safety:
// - Protected by write mutex (gp.mu)
// - Safe for concurrent access from multiple goroutines
// - Blocks other Updates and SetSigma operations while running
// - Blocks Predict operations while running
//
// Best practices:
// - Choose sigma based on expected smoothness of function
// - Consider validating sigma > 0 before calling
// - May need tuning for different optimization problems.
func (gp *gaussianProcess) SetSigma(sigma float64) {
	gp.mu.Lock()
	defer gp.mu.Unlock()
	gp.sigma = sigma
}

// GetSigma returns the current kernel width parameter (sigma) of the Gaussian Process.
// This value determines how quickly the influence of observations decreases with
// distance.
//
// Returns:
// - float64: Current sigma value
//
// Usage example:
//
//	gp := newGaussianProcess()
//
//	// Get current sigma value
//	currentSigma := gp.GetSigma()
//	fmt.Printf("Current kernel width: %f\n", currentSigma)
//
//	// Use sigma value for calculations
//	if gp.GetSigma() < 1.0 {
//	    // Handle narrow kernel case
//	}
//
// Important notes:
// - Uses read lock for better concurrency
// - Returns copy of sigma (safe to modify)
// - Default value is 1.0 (set in newGaussianProcess)
//
// Thread safety:
// - Protected by read mutex (gp.mu)
// - Multiple concurrent reads allowed
// - Blocked by Update and SetSigma operations
// - Safe for concurrent access from multiple goroutines
//
// Performance considerations:
// - Very fast operation (simple read)
// - Minimal locking impact on concurrent operations
// - No memory allocation.
func (gp *gaussianProcess) GetSigma() float64 {
	gp.mu.RLock()
	defer gp.mu.RUnlock()

	return gp.sigma
}

//////
// Factory.
//////

// newGaussianProcess creates and initializes a new Gaussian Process model
// with default parameters suitable for most optimization tasks.
//
// Returns:
// - *gaussianProcess: Pointer to newly initialized model
//
// Usage example:
//
//	gp := newGaussianProcess()
//	// Model ready for use with default sigma = 1.0
//
// Important notes:
// - Initializes with sigma = 1.0 (suitable for normalized inputs)
// - X and Y start empty (no observations)
// - Thread-safe from creation
//
// Best practices:
// - Create new instance for each optimization task
// - Consider adjusting sigma based on input scale
// - Don't share instances between independent optimizations.
func newGaussianProcess() *gaussianProcess {
	return &gaussianProcess{
		sigma: 1.0, // Default kernel width
	}
}
