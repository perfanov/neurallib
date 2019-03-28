/*
	Interface for a neural layer class. Contains necessary methods for training and computing a simple backpropagation network.
*/

#pragma once

#include <future>

#include "neuraldata.tcc"

template<class ScalarType>
class NeuronLayer
{
	// We want to not fill the call stack too much, so we don't want recursion. We'll try this by overloading with a lot of sleeping threads.
	// These threads will wake up as their compute input data becomes available.
	// In the future, it could be done with a thread pool - carousel.

	// The execution model is envisioned to be the following:
	// 1. A new session is created; all depended on futures are cleared.
	// 2. Each neuron layer instance will prepare a new future object.
	// Each layer's future will depend on other layers' futu
	public:
		//  TODO: shared_future? if a layer is depended on by two or more other layers.
		virtual std::future<NeuralData<ScalarType>> compute(NeuralData<ScalarType> const &) const = 0;
		// Returns the compute function for the previous neural layer.
		virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredCompute() const = 0;
		// Computes the gradient operator, that could be used for training,
		//virtual NeuralData<ScalarType>& gradient(NeuralData<ScalarType> const &) const = 0;
		// Returns the gradient operator for the next layer.
		//virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredGradient() const = 0;
	private:
		std::vector<std::weak_ptr<NeuronLayer<ScalarType>>> _attachedInputLayers;
		std::vector<std::weak_ptr<NeuronLayer<ScalarType>>> _attachedOutputLayers;
		void attachInput(std::shared_ptr<NeuronLayer<ScalarType>> const & otherLayer) final { _attachedInputLayers.push_back(otherLayer); }
		void attachOutput(std::shared_ptr<NeuronLayer<ScalarType>> const & otherLayer) final { _attachedOutputLayers.push_back(otherLayer); }
};