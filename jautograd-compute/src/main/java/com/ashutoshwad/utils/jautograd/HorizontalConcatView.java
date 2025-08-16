package com.ashutoshwad.utils.jautograd;

import java.util.Arrays;
import java.util.Objects;

public class HorizontalConcatView extends Matrix  {
    private final Matrix[]matrices;
    private final int numCols;

    public HorizontalConcatView(Matrix... matrices) {
        super(Arrays.stream(matrices).map(Matrix::getRequiresGradient).reduce((r1, r2)->r1||r2).orElse(false),
                combineForwardComputations(matrices),
                combineBackwardComputations(matrices));
        this.matrices = Objects.requireNonNull(matrices);
        if(matrices.length == 0) {
            throw new IllegalArgumentException("Hey there! why do you want to concatinate nothing? Avoiding creation of a black hole.");
        }
        int temp = matrices[0].numCols();
        int rows = matrices[0].numRows();
        for (int i = 1; i < matrices.length; i++) {
            if(rows != matrices[i].numRows()) {
                throw new IllegalArgumentException("Only matrices with the same number of rows can be horizontally concatinated!");
            }
            temp += matrices[i].numCols();
        }
        numCols = temp;
    }

    private static ForwardComputeOperation combineForwardComputations(Matrix[] matrices) {
        ForwardComputeOperation[]forwardComputeOps = new ForwardComputeOperation[matrices.length];
        for (int i = 0; i < matrices.length; i++) {
            forwardComputeOps[i] = matrices[i].forwardComputeOperation;
        }
        return new ManyToOneForwardComputeOperation(forwardComputeOps);
    }

    private static BackwardComputeOperation combineBackwardComputations(Matrix[] matrices) {
        BackwardComputeOperation[]backwardOps = new BackwardComputeOperation[matrices.length];
        for (int i = 0; i < matrices.length; i++) {
            backwardOps[i] = matrices[i].backwardComputeOperation;
        }
        return new ManyToOneBackwardComputeOperation(backwardOps);
    }

    private record IndexTuple(int matrixNum, int columnIndex){}
    private IndexTuple mapCols(final int index) {
        int matrixNum = 0;
        int colIndex = index;
        for (Matrix matrix : matrices) {
            final int currentNumCols = matrix.numCols();
            if (colIndex >= currentNumCols) {
                colIndex = colIndex - currentNumCols;
                matrixNum++;
            } else {
                return new IndexTuple(matrixNum, colIndex);
            }
        }
        throw new ArrayIndexOutOfBoundsException("This matrix has " + numCols + " columns, but an attempt was made to access index: " + index);
    }

    // Accessor methods
    public double getValue(int row, int column) {
        IndexTuple tup = mapCols(column);
        return matrices[tup.matrixNum].getValue(row, tup.columnIndex);
    }
    public synchronized void setValue(int row, int column, double value) {
        IndexTuple tup = mapCols(column);
        matrices[tup.matrixNum].setValue(row, tup.columnIndex, value);
    }
    public double getGradient(int row, int column) {
        IndexTuple tup = mapCols(column);
        return matrices[tup.matrixNum].getGradient(row, tup.columnIndex);
    }
    public synchronized void setGradient(int row, int column, double value) {
        IndexTuple tup = mapCols(column);
        matrices[tup.matrixNum].setGradient(row, tup.columnIndex, value);
    }
    public synchronized void accumulateGradient(int row, int column, double value) {
        IndexTuple tup = mapCols(column);
        matrices[tup.matrixNum].accumulateGradient(row, tup.columnIndex, value);
    }
    public int numRows() {
        return matrices[0].numRows();
    }
    public int numCols() {
        return numCols;
    }
}
