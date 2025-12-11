package com.ashutoshwad.utils.jautograd.matmulkernels;

import com.ashutoshwad.utils.jautograd.MatrixStore;

/**
 * Contract for high-performance matrix multiplication kernels.
 *
 * <p>This kernel computes a standard matrix product of the form:
 *
 * <pre>
 *   result = op(left) × op(right)
 * </pre>
 *
 * where:
 * <ul>
 *   <li>{@code op(left)} is either {@code left} or its logical transpose,</li>
 *   <li>{@code op(right)} is either {@code right} or its logical transpose,</li>
 *   <li>No transpose of {@code result} is supported.</li>
 * </ul>
 */
public interface MatrixMultiplicationKernel {
    /**
     * Computes:
     *
     * <pre>
     *   result = op(left) × op(right)
     * </pre>
     *
     * where {@code op(X)} is either {@code X} or its logical transpose.
     *
     * @throws IllegalArgumentException if dimensions are incompatible
     * @throws NullPointerException     if any argument is null
     */
    void matmul(
            MatrixStore left,
            boolean leftTranspose,
            MatrixStore right,
            boolean rightTranspose,
            MatrixStore result
    );

    /**
     * Computes:
     *
     * <pre>
     *   result = left × right
     * </pre>
     */
    default void matmul(MatrixStore left, MatrixStore right, MatrixStore result) {
        matmul(left, false, right, false, result);
    }

    /**
     * Computes:
     *
     * <pre>
     *   result = leftᵀ × right
     * </pre>
     */
    default void matmulTransposeLeft(MatrixStore left, MatrixStore right, MatrixStore result) {
        matmul(left, true, right, false, result);
    }

    /**
     * Computes:
     *
     * <pre>
     *   result = left × rightᵀ
     * </pre>
     */
    default void matmulTransposeRight(MatrixStore left, MatrixStore right, MatrixStore result) {
        matmul(left, false, right, true, result);
    }

    /**
     * Computes:
     *
     * <pre>
     *   result = leftᵀ × rightᵀ
     * </pre>
     */
    default void matmulTransposeBothInputs(MatrixStore left, MatrixStore right, MatrixStore result) {
        matmul(left, true, right, true, result);
    }
}
