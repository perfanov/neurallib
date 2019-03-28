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

TEST(SimpleNeurons, ConstantNeuron) {

}