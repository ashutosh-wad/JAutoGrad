package com.ashutoshwad.utils.jautograd.matmulkernels;

public class TiledMatrixMultiplicationKernelOptimizedTest extends AbstractMatrixMultiplicationKernelTest {
    @Override
    protected MatrixMultiplicationKernel createKernel() {
        return new TiledMatrixMultiplicationKernelOptimized();
    }
}
