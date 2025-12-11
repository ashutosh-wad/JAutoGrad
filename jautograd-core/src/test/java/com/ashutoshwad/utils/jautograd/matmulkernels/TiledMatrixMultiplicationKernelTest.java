package com.ashutoshwad.utils.jautograd.matmulkernels;

public class TiledMatrixMultiplicationKernelTest extends AbstractMatrixMultiplicationKernelTest {
    @Override
    protected MatrixMultiplicationKernel createKernel() {
        return new TiledMatrixMultiplicationKernel();
    }
}
