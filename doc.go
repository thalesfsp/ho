// Package ho provides automated hyperparameter optimization using Bayesian optimization
// with Gaussian Processes. It offers efficient, thread-safe optimization capabilities
// for tuning system parameters with minimal manual intervention.
//
// # Features
//
// The package includes the following key features:
//
//   - Bayesian Optimization: Uses Gaussian Process regression to efficiently explore
//     parameter spaces
//   - Thread-safe Implementation: All components are designed for concurrent
//     optimization runs
//   - Multiple Acquisition Functions: Various strategies for parameter space
//     exploration including Upper Confidence Bound (UCB), Probability of
//     Improvement (PI), Expected Improvement (EI), and Thompson Sampling
//   - Generic Implementation: Works with both integer and floating-point parameters
//   - Progress Monitoring: Real-time updates on optimization progress via channels
//   - Flexible Configuration: Highly customizable optimization process
//   - Automatic Parameter Tuning: Learns from previous evaluations to suggest
//     better parameters
//   - Robust Error Handling: Comprehensive error handling for benchmark functions
//
// # Installation
//
// To install the package, use:
//
//	go get github.com/yourusername/ho
//
// # Acquisition Functions
//
// The library provides four acquisition functions for different optimization strategies:
//
// 1. Upper Confidence Bound (UCB):
//
//   - Balances exploration and exploitation
//
//   - Controlled by Beta parameter (higher = more exploration)
//
//   - Default choice, works well in most cases
//
//     config := DefaultConfig()  // Uses UCB by default
//     config.AcqParams.Beta = 2.0  // Adjust exploration-exploitation trade-off
//
// 2. Probability of Improvement (PI):
//
//   - Conservative exploration strategy
//
//   - Focuses on small, reliable improvements
//
//   - Good for noise-sensitive applications
//
//     config := DefaultConfig()
//     config.AcquisitionFunc = ProbabilityOfImprovement
//     config.AcqParams.Xi = 0.01  // Minimum improvement threshold
//
// 3. Expected Improvement (EI):
//
//   - Balances improvement probability and magnitude
//
//   - Most commonly used in practice
//
//   - Good for general optimization tasks
//
//     config := DefaultConfig()
//     config.AcquisitionFunc = ExpectedImprovement
//     config.AcqParams.Xi = 0.01  // Minimum improvement threshold
//
// 4. Thompson Sampling:
//
//   - Simple but effective random sampling approach
//
//   - Great for parallel optimization
//
//   - No parameter tuning required
//
//     config := DefaultConfig()
//     config.AcquisitionFunc = ThompsonSampling
//     config.AcqParams.RandomState = rand.New(rand.NewSource(time.Now().UnixNano()))
//
// # Configuration
//
// The OptimizationConfig struct allows customization of the optimization process:
//
//	type OptimizationConfig struct {
//	    Iterations      int              // Number of optimization steps
//	    InitialSamples  int              // Initial random samples
//	    NumCandidates   int              // Candidates per iteration
//	    AcquisitionFunc AcquisitionFunc  // Strategy for point selection
//	    AcqParams       AcquisitionParams // Parameters for acquisition function
//	    ProgressChan    chan<- ProgressUpdate // For progress monitoring
//	}
//
// Recommended settings:
//   - Iterations: 20-200 (more = better results but longer runtime)
//   - InitialSamples: 5-20 (more = better initial model)
//   - NumCandidates: 50-500 (more = better search but slower iterations)
//
// # Thread Safety
//
// All components are designed to be thread-safe:
//   - Safe for concurrent optimization runs with different configs
//   - Gaussian Process model uses RWMutex for thread-safe updates
//   - Progress channel updates are properly synchronized
//   - Random number generation is thread-safe
//
// # Contributing
//
// To contribute to the project:
//  1. Fork the repository
//  2. Clone your fork
//  3. Create a feature branch
//  4. Make your changes
//  5. Run tests
//  6. Create a pull request
package ho
