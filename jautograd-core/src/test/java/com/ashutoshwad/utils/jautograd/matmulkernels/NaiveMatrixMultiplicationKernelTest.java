package com.ashutoshwad.utils.jautograd.matmulkernels;

public class NaiveMatrixMultiplicationKernelTest extends AbstractMatrixMultiplicationKernelTest {
    @Override
    protected MatrixMultiplicationKernel createKernel() {
        return new NaiveMatrixMultiplicationKernel();
    }
}
