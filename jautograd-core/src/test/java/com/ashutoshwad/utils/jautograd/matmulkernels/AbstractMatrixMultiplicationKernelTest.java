package com.ashutoshwad.utils.jautograd.matmulkernels;


import com.ashutoshwad.utils.jautograd.MatrixStore;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests generated with ChatGPT
 */
public abstract class AbstractMatrixMultiplicationKernelTest {

    protected abstract MatrixMultiplicationKernel createKernel();

    private static final float EPS = 1e-4f;

    private static MatrixStore randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        MatrixStore m = new MatrixStore(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m.set(r, c, rnd.nextFloat() * 2f - 1f); // [-1, 1]
            }
        }
        return m;
    }

    private static MatrixStore referenceMatmul(MatrixStore left,
                                               boolean leftTranspose,
                                               MatrixStore right,
                                               boolean rightTranspose) {
        int lRows = left.numRows();
        int lCols = left.numCols();
        int rRows = right.numRows();
        int rCols = right.numCols();

        final int resRows;
        final int resCols;
        final int commonDim;

        if (leftTranspose && rightTranspose) {
            // (lCols x lRows) x (rCols x rRows)
            if (lRows != rCols) throw new IllegalArgumentException("incompatible");
            resRows = lCols;
            resCols = rRows;
            commonDim = lRows;
        } else if (!leftTranspose && rightTranspose) {
            // (lRows x lCols) x (rCols x rRows)
            if (lCols != rCols) throw new IllegalArgumentException("incompatible");
            resRows = lRows;
            resCols = rRows;
            commonDim = lCols;
        } else if (leftTranspose && !rightTranspose) {
            // (lCols x lRows) x (rRows x rCols)
            if (lRows != rRows) throw new IllegalArgumentException("incompatible");
            resRows = lCols;
            resCols = rCols;
            commonDim = lRows;
        } else {
            // (!leftTranspose && !rightTranspose)
            // (lRows x lCols) x (rRows x rCols)
            if (lCols != rRows) throw new IllegalArgumentException("incompatible");
            resRows = lRows;
            resCols = rCols;
            commonDim = lCols;
        }

        MatrixStore result = new MatrixStore(resRows, resCols);

        for (int i = 0; i < resRows; i++) {
            for (int j = 0; j < resCols; j++) {
                float acc = 0f;
                for (int k = 0; k < commonDim; k++) {
                    float a = leftTranspose ? left.get(k, i) : left.get(i, k);
                    float b = rightTranspose ? right.get(j, k) : right.get(k, j);
                    acc += a * b;
                }
                result.set(i, j, acc);
            }
        }
        return result;
    }

    private static void assertMatrixEquals(MatrixStore expected, MatrixStore actual, float eps) {
        assertEquals(expected.numRows(), actual.numRows(), "row count mismatch");
        assertEquals(expected.numCols(), actual.numCols(), "col count mismatch");
        for (int r = 0; r < expected.numRows(); r++) {
            for (int c = 0; c < expected.numCols(); c++) {
                float e = expected.get(r, c);
                float a = actual.get(r, c);
                if (Math.abs(e - a) > eps) {
                    fail("Mismatch at (" + r + "," + c + "): expected=" + e + " actual=" + a);
                }
            }
        }
    }

    @Test
    void testSmallKnownValues_noTranspose() {
        MatrixMultiplicationKernel kernel = createKernel();

        MatrixStore A = new MatrixStore(2, 3);
        MatrixStore B = new MatrixStore(3, 2);

        // A = [1 2 3; 4 5 6]
        int v = 1;
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                A.set(r, c, v++);
            }
        }

        // B = [7 8; 9 10; 11 12]
        v = 7;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 2; c++) {
                B.set(r, c, v++);
            }
        }

        MatrixStore result = new MatrixStore(2, 2);
        kernel.matmul(A, B, result);

        /*
         * Expected:
         * [ 58  64 ]
         * [139 154]
         */
        assertEquals(58f, result.get(0, 0), EPS);
        assertEquals(64f, result.get(0, 1), EPS);
        assertEquals(139f, result.get(1, 0), EPS);
        assertEquals(154f, result.get(1, 1), EPS);
    }

    @Test
    void testTransposeVariants_small() {
        MatrixMultiplicationKernel kernel = createKernel();

        // Use square shapes so all 4 transpose combos are valid
        MatrixStore A = randomMatrix(3, 3, 123);
        MatrixStore B = randomMatrix(3, 3, 456);

        // 1) no transpose: A (3x3) × B (3x3) → (3x3)
        {
            MatrixStore expected = referenceMatmul(A, false, B, false);
            MatrixStore actual   = new MatrixStore(3, 3);
            kernel.matmul(A, B, actual);
            assertMatrixEquals(expected, actual, EPS);
        }

        // 2) leftᵀ × right: Aᵀ (3x3) × B (3x3) → (3x3)
        {
            MatrixStore expected = referenceMatmul(A, true, B, false);
            MatrixStore actual   = new MatrixStore(3, 3);
            kernel.matmulTransposeLeft(A, B, actual);
            assertMatrixEquals(expected, actual, EPS);
        }

        // 3) left × rightᵀ: A (3x3) × Bᵀ (3x3) → (3x3)
        {
            MatrixStore expected = referenceMatmul(A, false, B, true);
            MatrixStore actual   = new MatrixStore(3, 3);
            kernel.matmulTransposeRight(A, B, actual);
            assertMatrixEquals(expected, actual, EPS);
        }

        // 4) leftᵀ × rightᵀ: Aᵀ (3x3) × Bᵀ (3x3) → (3x3)
        {
            MatrixStore expected = referenceMatmul(A, true, B, true);
            MatrixStore actual   = new MatrixStore(3, 3);
            kernel.matmulTransposeBothInputs(A, B, actual);
            assertMatrixEquals(expected, actual, EPS);
        }
    }

    @Test
    void testRandomShapes_allTransposeCombos() {
        MatrixMultiplicationKernel kernel = createKernel();
        long seed = 999;
        Random rnd = new Random(seed);

        int[] dims = {1, 2, 3, 5, 8};

        for (int lRows : dims) {
            for (int lCols : dims) {
                for (int rRows : dims) {
                    for (int rCols : dims) {

                        MatrixStore L = randomMatrix(lRows, lCols, rnd.nextLong());
                        MatrixStore R = randomMatrix(rRows, rCols, rnd.nextLong());

                        boolean[] flags = {false, true};

                        for (boolean lt : flags) {
                            for (boolean rt : flags) {
                                // Check compatibility and skip if invalid
                                boolean compatible;
                                final int resRows, resCols, commonDim;

                                if (lt && rt) {
                                    compatible = (lRows == rCols);
                                    resRows = lCols;
                                    resCols = rRows;
                                    commonDim = lRows;
                                } else if (!lt && rt) {
                                    compatible = (lCols == rCols);
                                    resRows = lRows;
                                    resCols = rRows;
                                    commonDim = lCols;
                                } else if (lt && !rt) {
                                    compatible = (lRows == rRows);
                                    resRows = lCols;
                                    resCols = rCols;
                                    commonDim = lRows;
                                } else {
                                    compatible = (lCols == rRows);
                                    resRows = lRows;
                                    resCols = rCols;
                                    commonDim = lCols;
                                }

                                if (!compatible || commonDim == 0) {
                                    continue;
                                }

                                MatrixStore expected = referenceMatmul(L, lt, R, rt);
                                MatrixStore actual = new MatrixStore(resRows, resCols);

                                kernel.matmul(L, lt, R, rt, actual);
                                assertMatrixEquals(expected, actual, EPS);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void testDimensionMismatchThrows() {
        MatrixMultiplicationKernel kernel = createKernel();

        MatrixStore A = new MatrixStore(2, 3);
        MatrixStore B = new MatrixStore(4, 5);
        MatrixStore res = new MatrixStore(2, 5);

        assertThrows(IllegalArgumentException.class,
                () -> kernel.matmul(A, false, B, false, res));
        assertThrows(IllegalArgumentException.class,
                () -> kernel.matmul(A, true, B, false, res));
        assertThrows(IllegalArgumentException.class,
                () -> kernel.matmul(A, false, B, true, res));
        assertThrows(IllegalArgumentException.class,
                () -> kernel.matmul(A, true, B, true, res));
    }

    @Test
    void testInputsAreNotModified() {
        MatrixMultiplicationKernel kernel = createKernel();

        MatrixStore A = randomMatrix(3, 4, 111);
        MatrixStore B = randomMatrix(4, 2, 222);

        float[] aBefore = A.backingArray().clone();
        float[] bBefore = B.backingArray().clone();

        MatrixStore res = new MatrixStore(3, 2);
        kernel.matmul(A, B, res);

        // ensure input backing arrays unchanged
        assertArrayEquals(aBefore, A.backingArray(), "left matrix modified");
        assertArrayEquals(bBefore, B.backingArray(), "right matrix modified");
    }

    @Test
    void multiTile_noTranspose_rectangular() {
        MatrixMultiplicationKernel kernel = createKernel();

        // Chosen to cross multiple 42x42 tiles in all dims
        int lRows = 42 * 2 + 5;   // 89
        int common = 42 * 3 + 7;  // 133
        int rCols = 42 * 2 + 11;  // 95

        MatrixStore A = randomMatrix(lRows, common, 1234L);
        MatrixStore B = randomMatrix(common, rCols, 5678L);

        MatrixStore expected = referenceMatmul(A, false, B, false);
        MatrixStore actual   = new MatrixStore(lRows, rCols);

        kernel.matmul(A, B, actual);

        assertMatrixEquals(expected, actual, 1e-4f);
    }

    @Test
    void multiTile_rightTranspose() {
        MatrixMultiplicationKernel kernel = createKernel();

        int m = 42 * 2 + 3;   // rows of A
        int n = 42 * 2 + 9;   // common dim (cols of A, cols of B)
        int k = 42 * 2 + 13;  // rows of B, cols of result

        // A: (m x n)
        // B: (k x n)
        MatrixStore A = randomMatrix(m, n, 3003L);
        MatrixStore B = randomMatrix(k, n, 4004L);

        // Expected: A × Bᵀ
        MatrixStore expected = referenceMatmul(A, false, B, true);
        MatrixStore actual   = new MatrixStore(m, k);

        kernel.matmulTransposeRight(A, B, actual);

        assertMatrixEquals(expected, actual, 1e-4f);
    }

    @Test
    void multiTile_bothTranspose() {
        MatrixMultiplicationKernel kernel = createKernel();

        int m = 42 * 2 + 4;   // common dim between Aᵀ and Bᵀ
        int n = 42 * 2 + 10;  // cols of A / rows of Aᵀ
        int k = 42 * 2 + 6;   // rows of B / cols of Bᵀ

        // A: (m x n)
        // B: (k x m)
        MatrixStore A = randomMatrix(m, n, 5005L);
        MatrixStore B = randomMatrix(k, m, 6006L);

        // Expected: Aᵀ × Bᵀ
        MatrixStore expected = referenceMatmul(A, true, B, true);
        MatrixStore actual   = new MatrixStore(n, k);

        kernel.matmulTransposeBothInputs(A, B, actual);

        assertMatrixEquals(expected, actual, 1e-4f);
    }
}