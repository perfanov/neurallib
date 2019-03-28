#include <iostream>

#include "gtest.h"
#include "neuraldata.h"
#include "simpleneurons.h"

TEST(GeneralTests, GTestSanity) {
	EXPECT_EQ(0, 0);
}

TEST(NeuralData, ZeroSizeAssert) {
	std::vector<unsigned> dimSizes = { 1, 0, 2, 4 };
	ASSERT_DEATH(
		{
			NeuralData<> myData(dimSizes);
		},
		".*"
	);
}

TEST(NeuralData, GetValue) {
	std::vector<unsigned> dimSizes = { 1, 3, 2, 4 };
	NeuralData<> myData(dimSizes);
	myData({0,2,0,0}) = 5;
	ASSERT_EQ(5, myData({0,2,0,0}));
}

TEST(NeuralData, GetNonexistantIndexAssert) {
	std::vector<unsigned> dimSizes = { 1, 3, 2, 4 };
	NeuralData<> myData(dimSizes);
	ASSERT_DEATH(
		{
			myData({0,0,2,0});
		},
		".*"
	);
}

TEST(NeuralData, DimensionsMismatchAssert) {
	std::vector<unsigned> dimSizes = { 1, 3, 2, 4 };
	NeuralData<> myData(dimSizes);

	ASSERT_DEATH(
		{
			myData({0,1,2});
		},
		".*"
	);
}

TEST(NeuralData, ReadWriteSimple)
{
	{
		std::vector<unsigned> dimSizes = { 5 };
		NeuralData<> myData(dimSizes);

		myData({ 3 }) = 3;
		ASSERT_EQ(3, myData({ 3 }));
	}

	{
		std::vector<unsigned> dimSizes = { 6, 6 };
		NeuralData<> myData(dimSizes);

		myData({ 3, 5 }) = 3;
		myData({ 5, 3 }) = 5;
		ASSERT_EQ(3, myData({ 3, 5 }));
		ASSERT_EQ(5, myData({ 5, 3 }));
	}
}


TEST(NeuralData, ReadWriteByCounting)
{
	// Make a simple counting algorithm. The indices of each dimension can be seen as bits - forming a byte ;)
	// We save the value that should correspond to each combination of indices, by counting.
	std::vector<unsigned> dimSizes = { 2, 2, 2, 2, 2, 2, 2, 2 };
	NeuralData<int> myData(dimSizes);

	std::vector<unsigned> curIndices = { 0, 0, 0, 0, 0, 0, 0, 0};
	int curValue = 0;
	myData({0,0,0,0,0,0,0,0}) = 0;
	while(true)
	{
		int dimIdx;
		for(dimIdx = dimSizes.size() - 1; curIndices[dimIdx] == dimSizes[dimIdx] - 1; --dimIdx)
		{
			if(dimIdx == -1)
				break;

			curIndices[dimIdx] = 0;
		}
		if(dimIdx == -1)
			break;

		++curIndices[dimIdx];

		// Set the value.
		myData(curIndices) = ++curValue;
		// Carriage return.
		dimIdx = dimSizes.size() - 1;
	}
	ASSERT_EQ(2, myData({0,0,0,0,0,0,1,0}));
	for(int i = 0 ; i < 256; ++i)
	{
		int usedByte = i;
		for(auto bitIdx = 7; bitIdx >= 0; --bitIdx)
		{
			curIndices[bitIdx] = usedByte & 1;
			usedByte >>= 1;
		}
		ASSERT_EQ(i, myData(curIndices));
	}
}

TEST(SimpleNeurons, ConstantNeuron)
{
	std::vector<unsigned> dimSizes = { 20 };
	NeuralData<char> myData(dimSizes);

	std::string inStr("ConstantNeuron");
	for(unsigned i = 0; i < inStr.size(); ++i)
		myData({ i }) = inStr[i];

	ConstantNeuron<char> singleNeuron(myData);
	auto outputPtr = singleNeuron.compute();
	ASSERT_NE(nullptr, outputPtr.get());
	ASSERT_EQ(myData, *outputPtr.get());
}