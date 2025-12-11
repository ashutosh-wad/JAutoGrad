package com.ashutoshwad.utils.jautograd.matmulkernels;

public class NaiveMultiThreadedMatrixMultiplicationKernelTest extends AbstractMatrixMultiplicationKernelTest {
    @Override
    protected MatrixMultiplicationKernel createKernel() {
        return new NaiveMultiThreadedMatrixMultiplicationKernel();
    }
}
