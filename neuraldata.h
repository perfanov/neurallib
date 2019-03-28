/*
	Defines inputs and outputs of computation, and neural parameters.
	That is, an n-dimensional tensor.
	It's probably a good idea to template this on the container type; and figure out how to specialize for batch data retrieval
*/

#pragma once

#include<vector>
#include<cstring>
#include<assert.h>

template<class ScalarType = double>
class NeuralData
{
	private:
		std::vector<unsigned> _dimSizes;
		std::vector<ScalarType> _buffer;

	public:
		NeuralData() { }
		NeuralData(std::vector<unsigned> const & dimensionSizes)
		{
			assert(dimensionSizes.size() > 0);

			int bufSize = 1;
			for(auto dimSize : dimensionSizes)
				bufSize *= dimSize;

			assert(bufSize > 0);
			_dimSizes = dimensionSizes;
			_buffer.resize(bufSize);
		}
		ScalarType & operator()(std::vector<unsigned> const & indeces)
		{
			assert(indeces.size() == _dimSizes.size());

			unsigned offset = 0;
			unsigned offsetStep = 1;
			auto index = indeces.cbegin();
			auto dimSize = _dimSizes.cbegin();

			// Lower index dimensions are the junior dimensions as layout in the bufffer.
			// So, the 0th dimension has a step of 1, dim1 has a step of (dim0 size), etc.
			for(; index != indeces.cend(); ++index)
			{
				assert(*index < *dimSize);
				offset += (*index) * offsetStep;
				offsetStep *= *dimSize;
				++dimSize;
			}

			assert(offset < _buffer.size() && offset >= 0);
			return _buffer[offset];
		}
		bool operator==(NeuralData<ScalarType> const & rhs) const
		{
			if(_dimSizes.size() != rhs._dimSizes.size())
				return false;
			if(_buffer.size() != rhs._buffer.size())
				return false;
			if(std::memcmp(&_dimSizes[0], &rhs._dimSizes[0], _dimSizes.size() * sizeof(unsigned)))
				return false;
			if(std::memcmp(&_buffer[0], &rhs._buffer[0], _buffer.size() * sizeof(ScalarType)))
				return false;

			return true;
		}
};