package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MultiplicationOperatorTransposeOptimizedMultiThread extends Matrix {
    private static final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private record SlicePair(float[]slice_one, float[]slice_two){}
    private static final ThreadLocal<SlicePair> localSlice = new ThreadLocal<>();
    private final ComputeNode left;
    private final ComputeNode right;
    public final String operatorName;

    public MultiplicationOperatorTransposeOptimizedMultiThread(ComputeNode left, ComputeNode right,
                                                               String operatorName) {
        super(validateAndCreateResultStore(left, right, operatorName), left, right);
        this.left = left;
        this.right = right;
        this.operatorName = operatorName;
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
        final MatrixStore values = getValues();
        final MatrixStore lValues = left.getValues();
        final MatrixStore rValues = right.getValues();

        final int oRows = values.numRows();
        final int oCols = values.numCols();
        final int lCols = lValues.numCols();
        final int rCols = rValues.numCols();
        final int commonDim = lValues.numCols();

        final float[] oBak = values.backingArray();
        //Zero the output matrix
        Arrays.fill(oBak, 0);
        final float[] lBak = lValues.backingArray();
        final float[] rBak = rValues.backingArray();

        final int TILE_SIZE = 42;
        final int TILE_WIDTH = TILE_SIZE;

        List<Future<?>>futures = new ArrayList<>(oRows);
        for (int oRow = 0; oRow < oRows; oRow+=TILE_SIZE) {
            final int oRowFinal = oRow;
            futures.add(executor.submit(()->{
                SlicePair pair = localSlice.get();
                if (null == pair) {
                    pair = new SlicePair(new float[TILE_SIZE*TILE_WIDTH], new float[TILE_SIZE*TILE_WIDTH]);
                    localSlice.set(pair);
                }
                final float[] slice_one = pair.slice_one;
                final float[] slice_two = pair.slice_two;
                for (int oCol = 0; oCol < oCols; oCol+=TILE_SIZE) {
                    int tileWidth = Math.min((oCols - oCol), TILE_SIZE);
                    int tileHeight = Math.min((oRows - oRowFinal), TILE_SIZE);

                    for (int cVal = 0; cVal < commonDim; cVal+=TILE_WIDTH) {
                        int cWidth = Math.min((commonDim - cVal), TILE_WIDTH);

                        //Left Matrix
                        for (int row = 0; row < tileHeight; row++) {
                            final int lRow = row + oRowFinal;
                            for (int col = 0; col < cWidth; col++) {
                                final int cCol = col + cVal;
                                slice_one[row*TILE_WIDTH+col] = lBak[lRow*lCols+cCol];
                            }
                        }

                        //Right Matrix
                        for (int row = 0; row < cWidth; row++) {
                            final int rRow = row + cVal;
                            for (int col = 0; col < tileWidth; col++) {
                                final int cCol = col + oCol;
                                slice_two[col*TILE_SIZE+row] = rBak[rRow*rCols+cCol];
                            }
                        }

                        //Matrix multiply for tiles here
                        for (int sliceRow = 0; sliceRow < tileHeight; sliceRow++) {
                            final int tempSliceRowOffset = sliceRow * TILE_WIDTH;
                            for (int sliceCol = 0; sliceCol < tileWidth; sliceCol++) {
                                final int tempSliceColOffset = sliceCol * TILE_SIZE;
                                float accumulator = 0;
                                for (int sliceCommonDim = 0; sliceCommonDim < cWidth; sliceCommonDim++) {
                                    accumulator += slice_one[tempSliceRowOffset + sliceCommonDim]
                                            * slice_two[tempSliceColOffset + sliceCommonDim];
                                }
                                oBak[(sliceRow+oRowFinal) * oCols + (sliceCol+oCol)] += accumulator;
                            }
                        }
                    }
                }
            }));
        }
        futures.forEach(f -> {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public void backpropogateGradients() {
        final boolean thisReq = requiresGradient();
        final boolean leftReq = left.requiresGradient();
        final boolean rightReq = right.requiresGradient();

        // If this node doesn't carry gradients or no parent needs them, bail out
        if (!thisReq || (!leftReq && !rightReq)) {
            return;
        }

        final MatrixStore lValues = left.getValues();     // A
        final MatrixStore rValues = right.getValues();    // B
        final MatrixStore oGradients = getGradients();    // dL/dC

        final float[] lBak = lValues.backingArray();
        final float[] rBak = rValues.backingArray();
        final float[] oGBak = oGradients.backingArray();

        final int m = getValues().numRows();   // rows of A, rows of C
        final int n = left.numCols();          // shared dim
        final int p = getValues().numCols();   // cols of B, cols of C

        // dL/dA = dL/dC @ B^T   (m×p @ p×n → m×n)
        if (leftReq) {
            MatrixStore lGradients = left.getGradients();
            float[] lGBak = lGradients.backingArray();

            for (int i = 0; i < m; i++) {
                for (int k = 0; k < n; k++) {
                    float sum = 0f;
                    for (int j = 0; j < p; j++) {
                        int oGIndex = oGradients.computeIndex(i, j); // dL/dC[i,j]
                        int rIndex = rValues.computeIndex(k, j);      // B[k,j] (B^T[j,k])
                        sum += oGBak[oGIndex] * rBak[rIndex];
                    }
                    int lGIndex = lGradients.computeIndex(i, k);      // dL/dA[i,k]
                    lGBak[lGIndex] += sum;
                }
            }
        }

        // dL/dB = A^T @ dL/dC   (n×m @ m×p → n×p)
        if (rightReq) {
            MatrixStore rGradients = right.getGradients();
            float[] rGBak = rGradients.backingArray();

            for (int k = 0; k < n; k++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0f;
                    for (int i = 0; i < m; i++) {
                        int lIndex = lValues.computeIndex(i, k);      // A[i,k] (A^T[k,i])
                        int oGIndex = oGradients.computeIndex(i, j);  // dL/dC[i,j]
                        sum += lBak[lIndex] * oGBak[oGIndex];
                    }
                    int rGIndex = rGradients.computeIndex(k, j);      // dL/dB[k,j]
                    rGBak[rGIndex] += sum;
                }
            }
        }
    }
}
