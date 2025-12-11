package com.ashutoshwad.utils.jautograd;

import com.ashutoshwad.utils.jautograd.matmulkernels.*;

public class MultiplicationOperator extends Matrix {
    private final ComputeNode left;
    private final ComputeNode right;
    public final String operatorName;
    private final MatrixMultiplicationKernel kernel;

    public MultiplicationOperator(ComputeNode left, ComputeNode right,
                                     String operatorName) {
        super(validateAndCreateResultStore(left, right, operatorName), left, right);
        this.left = left;
        this.right = right;
        this.operatorName = operatorName;
        long flops = left.numRows() * left.numCols() * right.numCols();
        if (flops >= 1000000) {
            System.out.println("Large, going with uber");
            this.kernel = new TiledMatrixMultiplicationKernel();
            //this.kernel = new NaiveMultiThreadedMatrixMultiplicationKernel();
        } else {
            System.out.println("Small, going with native");
            this.kernel = new NaiveMatrixMultiplicationKernel();
        }
        computeResult();
    }

    private record Dimension(int numRows, int numCols){}

    private static MatrixStore validateAndCreateResultStore(ComputeNode left, ComputeNode right, String operatorName) {
        if (left.numCols() != right.numRows()) {
            throw new IllegalArgumentException("When multiplying two matrices, the number of columns of the first must equal the number of rows of the second. " +
                    "Found ("+left.numRows()+", "+left.numCols()+") & ("+right.numRows()+", "+right.numCols()+")");
        }
        return new MatrixStore(left.numRows(), right.numCols());
    }

    @Override
    public void computeResult() {
        kernel.matmul(left.getValues(), right.getValues(), getValues());
    }

    @Override
    public void backpropogateGradients() {
        if (!requiresGradient()) {
            return;
        }
        if (left.requiresGradient()) {
            kernel.matmulTransposeRight(getGradients(), right.getValues(), left.getGradients());
        }
        if (right.requiresGradient()) {
            kernel.matmulTransposeLeft(left.getValues(), getGradients(), right.getGradients());
        }
    }
}
