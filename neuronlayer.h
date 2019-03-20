/*
	Interface for a neural layer class. Contains necessary methods for training and computing a simple backpropagation network.
*/

#pragma once

#include <future>

#include "neuraldata.tcc"

template<class ScalarType>
class NeuralLayer
{
	// NOTE: we have to figure out if the NeuralLayer will have visibility over its previously computed results or gradients.
	// For example, it could use a shared_ptr; although this may cause thread safety problems.
	public:
		virtual std::future<NeuralData<ScalarType>> compute(NeuralData<ScalarType> const &) const = 0;
		// Returns the compute function for the previous neural layer.
		virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredCompute() const = 0;
		// Computes the gradient operator, that could be used for training,
		virtual NeuralData<ScalarType>& gradient(NeuralData<ScalarType> const &) const = 0;
		// Returns the gradient operator for the next layer.
		virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredGradient() const = 0;
	private:
		std::vector<std::weak_ptr<NeuralLayer<ScalarType>>> _attachedInputLayers;
		std::vector<std::weak_ptr<NeuralLayer<ScalarType>>> _attachedOutputLayers;
		void attachInput(std::shared_ptr<NeuralLayer<ScalarType>> const & otherLayer) { _attachedInputLayers.push_back(otherLayer); }
		void attachOutput(std::shared_ptr<NeuralLayer<ScalarType>> const & otherLayer) { _attachedOutputLayers.push_back(otherLayer); }
};