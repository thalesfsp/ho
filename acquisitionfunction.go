package ho

import "math"

//////
// Available acquisition functions for Bayesian optimization.
// Each function helps decide which points to evaluate next by balancing
// exploration (trying new areas) and exploitation (focusing on known good areas).
//////

// UCB implements the Upper Confidence Bound acquisition function.
//
// How it works:
// - Combines the predicted mean performance with the uncertainty (variance)
// - Lower values are better (we're minimizing execution time)
// - The Beta parameter controls the trade-off between exploration and exploitation
//
// Parameters:
// - mean: Predicted performance at this point
// - variance: Uncertainty in the prediction
// - params.Beta: Exploration weight (higher = more exploration)
//
// When to use:
// - General purpose, works well in most cases
// - When you want direct control over exploration-exploitation trade-off
// - When you need a simple, robust approach
//
// Example:
//
//	params := AcquisitionParams{
//	    Beta: 2.0,  // Balance between exploration and exploitation
//	}
//	value := UCB(0.5, 0.2, params)  // Evaluate a point with mean=0.5, variance=0.2
func UCB(mean, variance float64, params AcquisitionParams) float64 {
	return mean - params.Beta*math.Sqrt(variance)
}

// ProbabilityOfImprovement (PI) calculates the probability that a point will
// improve upon the current best observed value.
//
// How it works:
// - Estimates the probability of finding a better value than our current best
// - Uses a normal distribution assumption
// - Xi parameter adds a minimum improvement requirement
//
// Parameters:
// - mean: Predicted performance at this point
// - variance: Uncertainty in the prediction
// - params.BestSoFar: Best value observed so far
// - params.Xi: Minimum improvement desired
//
// When to use:
// - When you want to be conservative in exploring new points
// - When you're fine with small improvements
// - In problems where being "probably better" is more important than "how much better"
//
// Example:
//
//	params := AcquisitionParams{
//	    BestSoFar: 1.0,  // Current best execution time
//	    Xi: 0.01,        // Look for at least 1% improvement
//	}
//	prob := ProbabilityOfImprovement(0.9, 0.2, params)
func ProbabilityOfImprovement(mean, variance float64, params AcquisitionParams) float64 {
	z := (mean - params.BestSoFar - params.Xi) / math.Sqrt(variance)

	return normalCDF(z)
}

// ExpectedImprovement (EI) calculates the expected value of the improvement
// over the current best value.
//
// How it works:
// - Combines the probability of improvement with the magnitude of improvement
// - Balances how likely and how large the improvement might be
// - Often provides better exploration than PI
//
// Parameters:
// - mean: Predicted performance at this point
// - variance: Uncertainty in the prediction
// - params.BestSoFar: Best value observed so far
// - params.Xi: Minimum improvement desired
//
// When to use:
// - Most commonly used acquisition function
// - When you want to balance the size and probability of improvement
// - When you need a more nuanced approach than PI
// - In problems where the magnitude of improvement matters
//
// Example:
//
//	params := AcquisitionParams{
//	    BestSoFar: 1.0,  // Current best execution time
//	    Xi: 0.01,        // Look for at least 1% improvement
//	}
//	expected := ExpectedImprovement(0.9, 0.2, params)
func ExpectedImprovement(mean, variance float64, params AcquisitionParams) float64 {
	sigma := math.Sqrt(variance)

	z := (mean - params.BestSoFar - params.Xi) / sigma

	return (mean-params.BestSoFar-params.Xi)*normalCDF(z) + sigma*normalPDF(z)
}

// ThompsonSampling implements Thompson Sampling acquisition by drawing random
// samples from the posterior distribution.
//
// How it works:
// - Takes random samples from our belief about the function's behavior
// - Naturally balances exploration and exploitation
// - Uses randomness to explore the space
//
// Parameters:
// - mean: Predicted performance at this point
// - variance: Uncertainty in the prediction
// - params.RandomState: Random number generator (required!)
//
// When to use:
// - When you want a simple but effective approach
// - When you're running parallel optimizations
// - When you want to avoid the complexity of tuning Beta or Xi
// - In problems where random exploration is acceptable
//
// Example:
//
//	params := AcquisitionParams{
//	    RandomState: rand.New(rand.NewSource(time.Now().UnixNano())),
//	}
//	sample := ThompsonSampling(0.9, 0.2, params)
//
// Warning:
// - Always initialize RandomState before using this function
// - Don't share RandomState between different optimization runs.
func ThompsonSampling(mean, variance float64, params AcquisitionParams) float64 {
	return mean + math.Sqrt(variance)*params.RandomState.NormFloat64()
}
