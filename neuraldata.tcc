/*
	Defines inputs and outputs of computation, and neural parameters.
	That is, an n-dimensional tensor.
	It's probably a good idea to template this on the container type; and figure out how to specialize for batch data retrieval
*/

#include<vector>
#include<assert.h>

template<class ScalarType = double>
class NeuralData
{
	private:
		std::vector<unsigned> _dimSizes;
		std::vector<ScalarType> _buffer;

	public:
		NeuralData() = delete;
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
		ScalarType & operator()(std::vector<int> const & indeces)
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

			return _buffer[offset];
		}
};