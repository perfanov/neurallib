/*
	Interface for a neural layer class. Contains necessary methods for training and computing a simple backpropagation network.
*/

#pragma once

#include <future>

#include "neuronlayer.h"

template<class ScalarType>
class ConstantNeuron : public NeuronLayer<ScalarType>
{
	public:
		ConstantNeuron() = delete;
		ConstantNeuron(NeuralData<ScalarType> const & data)
		{
			NeuronLayer<ScalarType>::setComputed(true);
			_data = std::make_shared<NeuralData<ScalarType>>(data);
		}

		virtual std::shared_ptr<NeuralData<ScalarType>> compute()
		{
			return _data;
		};
		// Returns the compute function for the previous neural layer.
		//virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredCompute() const = 0;
		// Computes the gradient operator, that could be used for training,
		//virtual NeuralData<ScalarType>& gradient(NeuralData<ScalarType> const &) const = 0;
		// Returns the gradient operator for the next layer.
		//virtual std::packaged_task<NeuralData<ScalarType>(NeuralData<ScalarType> const &)> & requiredGradient() const = 0;
	private:
		std::shared_ptr<NeuralData<ScalarType>> _data;
};