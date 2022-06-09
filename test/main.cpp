#include "tst_colourdifference.h"
#include "tst_cudakernel.h"
#include "tst_imagelibrary.h"
#include "tst_CUDAImageLibrary.h"
#include "tst_cellshape.h"
#include "tst_generator.h"

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
