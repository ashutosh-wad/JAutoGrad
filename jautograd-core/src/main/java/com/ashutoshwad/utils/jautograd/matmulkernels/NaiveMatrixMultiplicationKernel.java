package com.ashutoshwad.utils.jautograd.matmulkernels;

import com.ashutoshwad.utils.jautograd.MatrixStore;

public class NaiveMatrixMultiplicationKernel implements MatrixMultiplicationKernel {
    @Override
    public void matmul(MatrixStore left, boolean leftTranspose, MatrixStore right, boolean rightTranspose, MatrixStore result) {
        final float[] oBak = result.backingArray();
        final float[] lBak = left.backingArray();
        final float[] rBak = right.backingArray();

        final int lCols = left.numCols();
        final int lRows = left.numRows();
        final int rCols = right.numCols();
        final int rRows = right.numRows();
        final int resCols = result.numCols();
        final int resRows = result.numRows();

        if (!leftTranspose && !rightTranspose) {
            if (lCols!= rRows) {
                throw new IllegalArgumentException("Incompatible matrices, unable to multiply");
            }
            if ( (lRows != resRows) || (rCols != resCols)) {
                throw new IllegalArgumentException("Result dimensions are not correct for a matrix multiply operation");
            }
            for (int oRow = 0; oRow < resRows; oRow++) {
                for (int oCol = 0; oCol < resCols; oCol++) {
                    int oIndex = result.computeIndex(oRow, oCol);
                    float accumulator = 0;
                    for (int cDim = 0; cDim < lCols; cDim++) {
                        int lIndex = left.computeIndex(oRow, cDim);
                        int rIndex = right.computeIndex(cDim, oCol);
                        accumulator += lBak[lIndex] * rBak[rIndex];
                    }
                    oBak[oIndex] = accumulator;
                }
            }
            return;
        }
        if (!leftTranspose && rightTranspose) {
            if (lCols!= rCols) {
                throw new IllegalArgumentException("Incompatible matrices, unable to multiply");
            }
            if ( (lRows != resRows) || (rRows != resCols)) {
                throw new IllegalArgumentException("Result dimensions are not correct for a matrix multiply operation");
            }
            for (int oRow = 0; oRow < resRows; oRow++) {
                for (int oCol = 0; oCol < resCols; oCol++) {
                    int oIndex = result.computeIndex(oRow, oCol);
                    float accumulator = 0;
                    for (int cDim = 0; cDim < lCols; cDim++) {
                        int lIndex = left.computeIndex(oRow, cDim);
                        int rIndex = right.computeIndex(oCol, cDim);
                        accumulator += lBak[lIndex] * rBak[rIndex];
                    }
                    oBak[oIndex] = accumulator;
                }
            }
            return;
        }
        if (leftTranspose && !rightTranspose) {
            if (lRows!= rRows) {
                throw new IllegalArgumentException("Incompatible matrices, unable to multiply");
            }
            if ( (lCols != resRows) || (rCols != resCols)) {
                throw new IllegalArgumentException("Result dimensions are not correct for a matrix multiply operation");
            }
            for (int oRow = 0; oRow < resRows; oRow++) {
                for (int oCol = 0; oCol < resCols; oCol++) {
                    int oIndex = result.computeIndex(oRow, oCol);
                    float accumulator = 0;
                    for (int cDim = 0; cDim < lRows; cDim++) {
                        int lIndex = left.computeIndex(cDim, oRow);
                        int rIndex = right.computeIndex(cDim, oCol);
                        accumulator += lBak[lIndex] * rBak[rIndex];
                    }
                    oBak[oIndex] = accumulator;
                }
            }
            return;
        }
        if (leftTranspose && rightTranspose) {
            if (lRows != rCols) {
                throw new IllegalArgumentException("Incompatible matrices, unable to multiply");
            }
            if ( (lCols != resRows) || (rRows != resCols)) {
                throw new IllegalArgumentException("Result dimensions are not correct for a matrix multiply operation");
            }
            for (int oRow = 0; oRow < resRows; oRow++) {
                for (int oCol = 0; oCol < resCols; oCol++) {
                    int oIndex = result.computeIndex(oRow, oCol);
                    float accumulator = 0;
                    for (int cDim = 0; cDim < lRows; cDim++) {
                        int lIndex = left.computeIndex(cDim, oRow);
                        int rIndex = right.computeIndex(oCol, cDim);
                        accumulator += lBak[lIndex] * rBak[rIndex];
                    }
                    oBak[oIndex] = accumulator;
                }
            }
        }
    }
}
