package com.ashutoshwad.utils.jautograd;

import java.util.Arrays;
import java.util.Objects;

public class VerticalConcatView extends Matrix  {
    private final Matrix[]matrices;
    private final int numRows;

    public VerticalConcatView(Matrix... matrices) {
        super(Arrays.stream(matrices).map(Matrix::getRequiresGradient).reduce((r1, r2)->r1||r2).orElse(false),
                combineForwardComputations(matrices),
                combineBackwardComputations(matrices));
        this.matrices = Objects.requireNonNull(matrices);
        if(matrices.length == 0) {
            throw new IllegalArgumentException("Hey there! why do you want to concatinate nothing? Avoiding creation of a black hole.");
        }
        int temp = matrices[0].numRows();
        int cols = matrices[0].numCols();
        for (int i = 1; i < matrices.length; i++) {
            if(cols != matrices[i].numCols()) {
                throw new IllegalArgumentException("Only matrices with the same number of columns can be vertically concatinated!");
            }
            temp += matrices[i].numRows();
        }
        numRows = temp;
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

    private record IndexTuple(int matrixNum, int rowIndex){}
    private IndexTuple mapRows(final int index) {
        int matrixNum = 0;
        int rowIndex = index;
        for (Matrix matrix : matrices) {
            final int currentNumRows = matrix.numRows();
            if (rowIndex >= currentNumRows) {
                rowIndex = rowIndex - currentNumRows;
                matrixNum++;
            } else {
                return new IndexTuple(matrixNum, rowIndex);
            }
        }
        throw new ArrayIndexOutOfBoundsException("This matrix has " + numRows + " columns, but an attempt was made to access index: " + index);
    }

    // Accessor methods
    public double getValue(int row, int column) {
        IndexTuple tup = mapRows(row);
        return matrices[tup.matrixNum].getValue(tup.rowIndex, column);
    }
    public synchronized void setValue(int row, int column, double value) {
        IndexTuple tup = mapRows(row);
        matrices[tup.matrixNum].setValue(tup.rowIndex, column, value);
    }
    public double getGradient(int row, int column) {
        IndexTuple tup = mapRows(row);
        return matrices[tup.matrixNum].getGradient(tup.rowIndex, column);
    }
    public synchronized void setGradient(int row, int column, double value) {
        IndexTuple tup = mapRows(row);
        matrices[tup.matrixNum].setGradient(tup.rowIndex, column, value);
    }
    public synchronized void accumulateGradient(int row, int column, double value) {
        IndexTuple tup = mapRows(row);
        matrices[tup.matrixNum].accumulateGradient(tup.rowIndex, column, value);
    }
    public int numRows() {
        return numRows;
    }
    public int numCols() {
        return matrices[0].numCols();
    }
}
