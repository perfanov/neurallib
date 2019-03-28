/*
	Simple neurons such as constant, increment/decrement, add/diff, etc.
*/

#pragma once

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
	private:
		std::shared_ptr<NeuralData<ScalarType>> _data;
};