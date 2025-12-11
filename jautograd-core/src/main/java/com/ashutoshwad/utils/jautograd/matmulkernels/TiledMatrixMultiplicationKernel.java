package com.ashutoshwad.utils.jautograd.matmulkernels;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.ExecutorUtils;
import com.ashutoshwad.utils.jautograd.MatrixStore;
import com.ashutoshwad.utils.jautograd.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class TiledMatrixMultiplicationKernel implements MatrixMultiplicationKernel {
    private record KernelSlicePair(float[]sliceOne, float[]sliceTwo, float[]sliceRes){};
    private final ThreadLocal<KernelSlicePair> kernelSlicePairThreadLocal;
    private final int TILE_SIZE;
    private final int TILE_COMMON_DIM_SIZE;

    public TiledMatrixMultiplicationKernel() {
        final String tileSizeStr = System.getenv("tiled.matmul.tilesize");
        final String tileCommonDimSizeStr = System.getenv("tiled.matmul.commondimsize");

        if (StringUtils.isBlank(tileSizeStr)) {
            TILE_SIZE = 42;
        } else {
            TILE_SIZE = Integer.parseInt(tileSizeStr);
        }

        if (StringUtils.isBlank(tileCommonDimSizeStr)) {
            TILE_COMMON_DIM_SIZE = 42;
        } else {
            TILE_COMMON_DIM_SIZE = Integer.parseInt(tileCommonDimSizeStr);
        }

        kernelSlicePairThreadLocal = ThreadLocal.withInitial(()->new KernelSlicePair(new float[TILE_SIZE*TILE_COMMON_DIM_SIZE], new float[TILE_SIZE*TILE_COMMON_DIM_SIZE], new float[TILE_SIZE*TILE_SIZE]));
    }

    public TiledMatrixMultiplicationKernel(int tileSize, int tileCommonDimSize) {
        TILE_SIZE = tileSize;
        TILE_COMMON_DIM_SIZE = tileCommonDimSize;

        kernelSlicePairThreadLocal = ThreadLocal.withInitial(()->new KernelSlicePair(new float[TILE_SIZE*TILE_COMMON_DIM_SIZE], new float[TILE_SIZE*TILE_COMMON_DIM_SIZE], new float[TILE_SIZE*TILE_SIZE]));
    }

    @Override
    public void matmul(MatrixStore left,
                       final boolean leftTranspose,
                       MatrixStore right,
                       final boolean rightTranspose,
                       MatrixStore result) {

        final int lCols = left.numCols();
        final int lRows = left.numRows();
        final int rCols = right.numCols();
        final int rRows = right.numRows();
        final int resCols = result.numCols();
        final int resRows = result.numRows();
        final int commonDim;

        final ExecutorFactory.Details DETAILS = ExecutorFactory.getDetails();
        final int NUM_THREADS = DETAILS.numThreads();
        final ExecutorService executorService = DETAILS.executorService();

        List<Future<?>> futures = new ArrayList<>(NUM_THREADS);

        final float[] resultArr = result.backingArray();

        if (leftTranspose && rightTranspose) {
            if (lRows != rCols) {
                throw new IllegalArgumentException("Matrices are incompatible and cannnot be multiplied. ("+lCols+" x "+lRows+") x ("+rCols+" x "+rRows+")");
            }
            if (resRows!=lCols || resCols!=rRows) {
                throw new IllegalArgumentException("Result matrix is of incorrect dimensions ("+resRows+" x "+resCols+"). Inputs are ("+lCols+" x "+lRows+") x ("+rCols+" x "+rRows+")");
            }
            commonDim = lRows;
        } else if (!leftTranspose && rightTranspose) {
            if (lCols != rCols) {
                throw new IllegalArgumentException("Matrices are incompatible and cannnot be multiplied. ("+lRows+" x "+lCols+") x ("+rCols+" x "+rRows+")");
            }
            if (resRows!=lRows || resCols!=rRows) {
                throw new IllegalArgumentException("Result matrix is of incorrect dimensions ("+resRows+" x "+resCols+"). Inputs are ("+lRows+" x "+lCols+") x ("+rCols+" x "+rRows+")");
            }
            commonDim = lCols;
        } else if (leftTranspose && !rightTranspose) {
            if (lRows != rRows) {
                throw new IllegalArgumentException("Matrices are incompatible and cannnot be multiplied. ("+lCols+" x "+lRows+") x ("+rRows+" x "+rCols+")");
            }
            if (resRows!=lCols || resCols!=rCols) {
                throw new IllegalArgumentException("Result matrix is of incorrect dimensions ("+resRows+" x "+resCols+"). Inputs are ("+lCols+" x "+lRows+") x ("+rRows+" x "+rCols+")");
            }
            commonDim = lRows;
        } else {
            if (lCols != rRows) {
                throw new IllegalArgumentException("Matrices are incompatible and cannnot be multiplied. ("+lRows+" x "+lCols+") x ("+rRows+" x "+rCols+")");
            }
            if (resRows!=lRows || resCols!=rCols) {
                throw new IllegalArgumentException("Result matrix is of incorrect dimensions ("+resRows+" x "+resCols+"). Inputs are ("+lRows+" x "+lCols+") x ("+rRows+" x "+rCols+")");
            }
            commonDim = lCols;
        }

        //Zero the result array
        for (int t = 0; t < NUM_THREADS; t++) {
            final int tCopy = t;
            futures.add(executorService.submit(()->{
                for (int i = tCopy; i < resultArr.length; i+=NUM_THREADS) {
                    resultArr[i] = 0;
                }
            }));
        }
        ExecutorUtils.awaitFutures(futures);
        futures.clear();

        for (int tIdx = 0; tIdx < NUM_THREADS; tIdx++) {
            final int START_OFFSET = tIdx * TILE_SIZE;
            futures.add(executorService.submit(()->{

                //Fetch the slice buffers
                KernelSlicePair pair = kernelSlicePairThreadLocal.get();
                final float[]slizeOne = pair.sliceOne;
                final float[]slizeTwo = pair.sliceTwo;
                final float[]slizeRes = pair.sliceRes;
                final MatrixStore sliceStoreOne = new MatrixStore(TILE_SIZE, TILE_COMMON_DIM_SIZE, slizeOne);
                final MatrixStore sliceStoreTwo = new MatrixStore(TILE_SIZE, TILE_COMMON_DIM_SIZE, slizeTwo);
                final MatrixStore sliceStoreRes = new MatrixStore(TILE_SIZE, TILE_SIZE, slizeRes);

                //Start tiling loop
                for (int oRow = START_OFFSET; oRow < resRows; oRow += TILE_SIZE * NUM_THREADS) {
                    final int tileNumRows = Math.min(TILE_SIZE, resRows - oRow);

                    for (int oCol = 0; oCol < resCols; oCol += TILE_SIZE) {
                        final int tileNumCols = Math.min(TILE_SIZE, resCols - oCol);

                        for (int cDimIdx = 0; cDimIdx < commonDim; cDimIdx+=TILE_COMMON_DIM_SIZE) {
                            final int cDimLimit = Math.min(TILE_COMMON_DIM_SIZE, commonDim - cDimIdx);

                            //Start copy to slices
                            for (int tRow = 0; tRow < tileNumRows; tRow++) {
                                for (int tilecDimIdx = 0; tilecDimIdx < cDimLimit; tilecDimIdx++) {
                                    final float temp;
                                    if (leftTranspose) {
                                        temp = left.get(tilecDimIdx + cDimIdx, tRow+oRow);
                                    } else {
                                        temp = left.get(tRow+oRow, tilecDimIdx + cDimIdx);
                                    }
                                    sliceStoreOne.set(tRow, tilecDimIdx, temp);
                                }
                            }

                            for (int tCol = 0; tCol < tileNumCols; tCol++) {
                                for (int tilecDimIdx = 0; tilecDimIdx < cDimLimit; tilecDimIdx++) {
                                    final float temp;
                                    if (rightTranspose) {
                                        temp = right.get(tCol+oCol, tilecDimIdx + cDimIdx);
                                    } else {
                                        temp = right.get(tilecDimIdx + cDimIdx, tCol+oCol);
                                    }
                                    sliceStoreTwo.set(tCol, tilecDimIdx, temp);
                                }
                            }

                            //Actual Tile Matmul
                            for (int tRow = 0; tRow < tileNumRows; tRow++) {
                                for (int tCol = 0; tCol < tileNumCols; tCol++) {
                                    float accumulator = 0;
                                    for (int tilecDimIdx = 0; tilecDimIdx < cDimLimit; tilecDimIdx++) {
                                        accumulator += sliceStoreOne.get(tRow, tilecDimIdx)*sliceStoreTwo.get(tCol, tilecDimIdx);
                                    }
                                    sliceStoreRes.set(tRow, tCol, accumulator);
                                }
                            }

                            //Copy result back
                            for (int tRow = 0; tRow < tileNumRows; tRow++) {
                                for (int tCol = 0; tCol < tileNumCols; tCol++) {
                                    float temp = sliceStoreRes.get(tRow, tCol);
                                    result.add(oRow+tRow, oCol+tCol, temp);
                                }
                            }
                        }
                    }
                }
            }));
        }
        ExecutorUtils.awaitFutures(futures);
        futures.clear();
    }
}
