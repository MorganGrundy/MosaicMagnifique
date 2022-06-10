#include "tst_ColourDifference.h"
#include "tst_CUDAKernel.h"
#include "tst_ImageLibrary.h"
#include "tst_CellShape.h"
#include "tst_Generator.h"
#include "tst_CUDAGenerator.h"

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
