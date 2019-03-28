/*
	Interface for a neural layer class. Contains necessary methods for training and computing a simple backpropagation network.
*/

#pragma once

#include<memory>

#include "neuraldata.h"

template<class ScalarType>
class NeuronLayer
{
	// The execution model is envisioned to be the following:
	// 1. A new session is created; all depended on futures are cleared.
	// 2. Each neuron layer finds out its depth and how many layers it counts on.
	// 3. A task agent will start calling neurons that have 0 unfulfilled dependencies, in order of depth.

	// As a hypothetical musing, this class could be inherited by wrappers of CuDNN,
	// as long as we are using a gpu-template specialization of NeuralData<GpuDouble>, with different accessors, custom construct/destruct etc.
	public:
		inline bool isComputed() { return _isComputed; }

		// If the neuron does not have enough data, it will return an empty shared_ptr.
		virtual std::shared_ptr<NeuralData<ScalarType>> compute() = 0;
		// Returns the compute function for the previous neural layer.
		//virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredCompute() const = 0;
		// Computes the gradient operator, that could be used for training,
		//virtual NeuralData<ScalarType>& gradient(NeuralData<ScalarType> const &) const = 0;
		// Returns the gradient operator for the next layer.
		//virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredGradient() const = 0;

		virtual void attachInput(std::shared_ptr<NeuronLayer<ScalarType>> const & otherLayer) final { _attachedInputLayers.push_back(otherLayer); }
		virtual void attachOutput(std::shared_ptr<NeuronLayer<ScalarType>> const & otherLayer) final { _attachedOutputLayers.push_back(otherLayer); }
	protected:
		bool _isComputed;
		void setComputed(bool computed) { _isComputed = computed; }
	private:
		std::vector<std::weak_ptr<NeuronLayer<ScalarType>>> _attachedInputLayers;
		std::vector<std::weak_ptr<NeuronLayer<ScalarType>>> _attachedOutputLayers;
};