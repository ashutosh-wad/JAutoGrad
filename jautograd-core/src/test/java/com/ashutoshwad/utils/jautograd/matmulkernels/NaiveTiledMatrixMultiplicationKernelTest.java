package com.ashutoshwad.utils.jautograd.matmulkernels;

public class NaiveTiledMatrixMultiplicationKernelTest extends AbstractMatrixMultiplicationKernelTest {
    @Override
    protected MatrixMultiplicationKernel createKernel() {
        return new NaiveTiledMatrixMultiplicationKernel();
    }
}
