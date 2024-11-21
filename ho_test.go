package ho

import (
	"math/rand"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Sample function to be benchmarked.
func testFuncInt(bufferSize int, multipler int) error {
	// var delay holds the delay time in milliseconds which is random from 100 to 300 milliseconds.
	delay := rand.Intn(100) + 50

	// Replace this with your actual function
	buffer := make([]int, 0, bufferSize)

	// Simulate some work
	for i := 0; i < 1000000; i++ {
		if len(buffer) == bufferSize {
			buffer = buffer[:0]
		}

		buffer = append(buffer, i*multipler)
	}

	time.Sleep(time.Duration(delay) * time.Millisecond)

	return nil
}

func TestOptimizeBufferSize(t *testing.T) {
	// Using default configuration (UCB)
	config := DefaultConfig()

	// Your benchmark function
	benchmarkFunc := func(params ...int) error {
		return testFuncInt(params[0], params[1])
	}

	// Hyperparameter ranges
	ranges := []ParameterRange[int]{
		{Min: 1, Max: 100},
		{Min: 1, Max: 3},
	}

	// Run optimization with chosen configuration
	optimalSize := OptimizeHyperparameters(
		config,
		benchmarkFunc,
		ranges...,
	)

	// Assert has `optimalSize` has two elements.
	assert.Len(t, optimalSize, 2)
}

func TestOptimizeBufferSizeChannel(t *testing.T) {
	// Create a configuration
	config := DefaultConfig()

	// The following isn't necessary, this is just exist for testing purposes.
	config.InitialSamples = 3

	// The following isn't necessary, this is just exist for testing purposes.
	config.Iterations = 5

	// Create a bidirectional channel for progress updates
	progressChan := make(chan ProgressUpdate, config.InitialSamples+config.Iterations)
	defer close(progressChan)

	// Assign the channel to config (will be automatically converted to send-only)
	config.ProgressChan = progressChan

	// This isn't necessary when collecting metrics. This just exist for testing
	// purposes.
	var counter int32

	// Start a goroutine to handle progress updates.
	go func() {
		for update := range progressChan {
			// Atomic updates counter
			atomic.AddInt32(&counter, int32(update.CurrentIteration))
		}
	}()

	// Define parameter ranges
	ranges := []ParameterRange[int]{
		{Min: 1024, Max: 1048576}, // Buffer size (1KB to 1MB).
		{Min: 1, Max: 32},         // Worker count.
	}

	// Run optimization.
	bestParams := OptimizeHyperparameters(
		config,
		func(params ...int) error {
			return testFuncInt(params[0], params[1])
		},
		ranges...,
	)

	// Ensure events where emitted.
	assert.Greater(t, atomic.LoadInt32(&counter), int32(0))

	// Ensure optimal parameters are returned.
	assert.Len(t, bestParams, 2)
}

// Sample function to be benchmarked.
func testFuncFloat(bufferSize float32, multiplier float32) error {
	// var delay holds the delay time in milliseconds which is random from 100 to 300 milliseconds.
	delay := rand.Intn(100) + 50

	// Replace this with your actual function
	buffer := []float32{}

	// Simulate some work.
	for i := 0; i < 1000000; i++ {
		if len(buffer) == int(bufferSize) {
			buffer = buffer[:0]
		}

		buffer = append(buffer, float32(i)*multiplier)
	}

	time.Sleep(time.Duration(delay) * time.Millisecond)

	return nil
}

func TestOptimizeBufferSizeFloat(t *testing.T) {
	// Using default configuration (UCB)
	config := DefaultConfig()

	// Your benchmark function with type conversion
	benchmarkFunc := func(params ...float32) error {
		return testFuncFloat(params[0], params[1])
	}

	// Hyperparameter ranges
	ranges := []ParameterRange[float32]{
		{Min: 1, Max: 100}, // Buffer size range
		{Min: 1, Max: 3},   // Multiplier range
	}

	// Run optimization with chosen configuration
	optimalSize := OptimizeHyperparameters[float32](
		config,
		benchmarkFunc,
		ranges...,
	)

	// Assert has `optimalSize` has two elements.
	assert.Len(t, optimalSize, 2)
}

func TestOptimizeBufferSizeChannelFloat(t *testing.T) {
	// Create a configuration
	config := DefaultConfig()

	// The following isn't necessary, this is just exist for testing purposes.
	config.InitialSamples = 3

	// The following isn't necessary, this is just exist for testing purposes.
	config.Iterations = 5

	// Create a bidirectional channel for progress updates
	progressChan := make(chan ProgressUpdate, config.InitialSamples+config.Iterations)
	defer close(progressChan)

	// Assign the channel to config (will be automatically converted to send-only)
	config.ProgressChan = progressChan

	// This isn't necessary when collecting metrics. This just exist for testing
	// purposes.
	var counter int32

	// Start a goroutine to handle progress updates.
	go func() {
		for update := range progressChan {
			// Atomic updates counter
			atomic.AddInt32(&counter, int32(update.CurrentIteration))
		}
	}()

	// Define parameter ranges.
	ranges := []ParameterRange[float32]{
		{Min: 1024.0, Max: 1048576.0}, // Buffer size (1KB to 1MB).
		{Min: 1.0, Max: 32.0},         // Worker count.
	}

	// Run optimization with float32 parameters
	bestParams := OptimizeHyperparameters[float32](
		config,
		func(params ...float32) error {
			return testFuncFloat(params[0], params[1])
		},
		ranges...,
	)

	// Ensure events where emitted.
	assert.Greater(t, atomic.LoadInt32(&counter), int32(0))

	// Ensure optimal parameters are returned.
	assert.Len(t, bestParams, 2)
}
