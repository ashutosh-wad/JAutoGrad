package com.ashutoshwad.utils.jautograd.compute.tensor;

import com.ashutoshwad.utils.jautograd.compute.ComputeNode;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleSupplier;

/**
 * Concrete implementation of the Tensor interface using stride-based indexing.
 *
 * <p>This implementation stores tensor data in a single flat array with stride information
 * for efficient multi-dimensional access. All operations create new tensors to maintain
 * immutability and compatibility with the autograd system.</p>
 */
public class TensorImpl implements Tensor {

    private final ComputeNode[] data;
    private final int[] shape;
    private final int[] strides;
    private final int size;
    private final int rank;

    // Default random generator for factory methods
    private static final Random defaultRandom = new Random();

    /**
     * Internal constructor for creating tensors with existing data and metadata.
     *
     * @param data the flat array of ComputeNodes
     * @param shape the dimensions of the tensor
     * @param strides the stride information for each dimension
     */
    private TensorImpl(ComputeNode[] data, int[] shape, int[] strides) {
        this.data = data.clone(); // Defensive copy
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.rank = shape.length;
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    }

    /**
     * Creates a tensor with the specified shape, initializing elements using the supplier.
     */
    private TensorImpl(int[] shape, DoubleSupplier valueSupplier) {
        this.shape = shape.clone();
        this.rank = shape.length;
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        this.strides = computeStrides(shape);
        this.data = new ComputeNode[size];

        // Initialize all elements
        for (int i = 0; i < size; i++) {
            this.data[i] = new ComputeNode(valueSupplier.getAsDouble());
        }
    }

    /**
     * Computes the stride array for a given shape.
     * Stride[i] = product of all dimensions after dimension i.
     */
    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /**
     * Converts multi-dimensional indices to flat array index using strides.
     */
    private int computeFlatIndex(int[] indices) {
        if (indices.length != rank) {
            throw new IllegalArgumentException("Number of indices (" + indices.length +
                    ") must match tensor rank (" + rank + ")");
        }

        int flatIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] +
                        " out of bounds for dimension " + i + " with size " + shape[i]);
            }
            flatIndex += indices[i] * strides[i];
        }
        return flatIndex;
    }

    // ================================
    // Interface Implementation
    // ================================

    @Override
    public int[] getShape() {
        return shape.clone();
    }

    @Override
    public int getRank() {
        return rank;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public ComputeNode get(int... indices) {
        return data[computeFlatIndex(indices)];
    }

    @Override
    public void set(ComputeNode value, int... indices) {
        data[computeFlatIndex(indices)] = value;
    }

    // ================================
    // Factory Methods
    // ================================

    public static Tensor zeros(int... shape) {
        validateShape(shape);
        return new TensorImpl(shape, () -> 0.0);
    }

    public static Tensor ones(int... shape) {
        validateShape(shape);
        return new TensorImpl(shape, () -> 1.0);
    }

    public static Tensor random(int... shape) {
        validateShape(shape);
        return new TensorImpl(shape, () -> defaultRandom.nextGaussian());
    }

    /**
     * Creates a tensor with custom initialization.
     */
    public static Tensor create(DoubleSupplier supplier, int... shape) {
        validateShape(shape);
        return new TensorImpl(shape, supplier);
    }

    /**
     * Creates a tensor from a 2D array of ComputeNodes.
     */
    public static Tensor from(ComputeNode[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            throw new IllegalArgumentException("Matrix cannot be null or empty");
        }

        int rows = matrix.length;
        int cols = matrix[0].length;

        // Validate consistent column count
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new IllegalArgumentException("Inconsistent row lengths");
            }
        }

        int[] shape = {rows, cols};
        ComputeNode[] data = new ComputeNode[rows * cols];
        int[] strides = computeStrides(shape);

        // Copy data in row-major order
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * strides[0] + j * strides[1]] = matrix[i][j];
            }
        }

        return new TensorImpl(data, shape, strides);
    }

    private static void validateShape(int[] shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be null or empty");
        }
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive, got: " +
                        Arrays.toString(shape));
            }
        }
    }

    // ================================
    // Shape Operations
    // ================================

    @Override
    public Tensor reshape(int... newShape) {
        validateShape(newShape);

        int newSize = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (newSize != size) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + size +
                    " to shape " + Arrays.toString(newShape) + " (size " + newSize + ")");
        }

        int[] newStrides = computeStrides(newShape);
        return new TensorImpl(data, newShape, newStrides);
    }

    @Override
    public Tensor transpose() {
        if (rank < 2) {
            return new TensorImpl(data, shape, strides); // Return copy for consistency
        }

        // For 2D: swap dimensions 0 and 1
        // For N-D: reverse all dimensions
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = rank - 1 - i;
        }
        return transpose(axes);
    }

    @Override
    public Tensor transpose(int... axes) {
        if (axes.length != rank) {
            throw new IllegalArgumentException("Number of axes must match tensor rank");
        }

        // Validate axes
        boolean[] used = new boolean[rank];
        for (int axis : axes) {
            if (axis < 0 || axis >= rank) {
                throw new IllegalArgumentException("Axis " + axis + " out of range for rank " + rank);
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Duplicate axis: " + axis);
            }
            used[axis] = true;
        }

        // Compute new shape and strides
        int[] newShape = new int[rank];
        int[] newStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }

        return new TensorImpl(data, newShape, newStrides);
    }

    // ================================
    // Mathematical Operations
    // ================================

    @Override
    public Tensor add(Tensor other) {
        return elementWiseOperation(other, ComputeNode::add);
    }

    @Override
    public Tensor subtract(Tensor other) {
        return elementWiseOperation(other, ComputeNode::sub);
    }

    @Override
    public Tensor multiply(Tensor other) {
        return elementWiseOperation(other, ComputeNode::mul);
    }

    @Override
    public Tensor divide(Tensor other) {
        return elementWiseOperation(other, ComputeNode::div);
    }

    /**
     * Generic element-wise operation helper.
     */
    private Tensor elementWiseOperation(Tensor other, BinaryOperation op) {
        if (!Arrays.equals(shape, other.getShape())) {
            throw new IllegalArgumentException("Tensor shapes must match for element-wise operations. " +
                    "Got " + Arrays.toString(shape) + " and " + Arrays.toString(other.getShape()));
        }

        ComputeNode[] resultData = new ComputeNode[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = op.apply(data[i], ((TensorImpl) other).data[i]);
        }

        return new TensorImpl(resultData, shape, strides);
    }

    @FunctionalInterface
    private interface BinaryOperation {
        ComputeNode apply(ComputeNode a, ComputeNode b);
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (rank < 2 || other.getRank() < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires at least 2D tensors");
        }

        int[] otherShape = other.getShape();

        // For 2D case: (m, k) x (k, n) -> (m, n)
        if (rank == 2 && other.getRank() == 2) {
            return matmul2D(other);
        }

        throw new UnsupportedOperationException("Batched matrix multiplication not yet implemented");
    }

    /**
     * 2D matrix multiplication implementation.
     */
    private Tensor matmul2D(Tensor other) {
        int[] otherShape = other.getShape();
        int m = shape[0];      // rows of this
        int k = shape[1];      // cols of this / rows of other
        int n = otherShape[1]; // cols of other

        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication: " +
                    Arrays.toString(shape) + " x " + Arrays.toString(otherShape));
        }

        ComputeNode[] resultData = new ComputeNode[m * n];
        TensorImpl otherImpl = (TensorImpl) other;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ComputeNode sum = new ComputeNode(0.0);
                for (int l = 0; l < k; l++) {
                    ComputeNode a = data[i * strides[0] + l * strides[1]];
                    ComputeNode b = otherImpl.data[l * otherImpl.strides[0] + j * otherImpl.strides[1]];
                    sum = sum.add(a.mul(b));
                }
                resultData[i * n + j] = sum;
            }
        }

        return new TensorImpl(resultData, new int[]{m, n}, computeStrides(new int[]{m, n}));
    }

    // ================================
    // Activation Functions
    // ================================

    @Override
    public Tensor relu() {
        return elementWiseOperation(ComputeNode::relu);
    }

    @Override
    public Tensor leakyRelu(double negativeSlope) {
        return elementWiseOperation(node -> node.leakyRelu(negativeSlope));
    }

    /**
     * Generic element-wise unary operation helper.
     */
    private Tensor elementWiseOperation(UnaryOperation op) {
        ComputeNode[] resultData = new ComputeNode[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = op.apply(data[i]);
        }
        return new TensorImpl(resultData, shape, strides);
    }

    @FunctionalInterface
    private interface UnaryOperation {
        ComputeNode apply(ComputeNode node);
    }

    @Override
    public Tensor softmax(int dim) {
        if (dim < 0 || dim >= rank) {
            throw new IllegalArgumentException("Dimension " + dim + " out of range for rank " + rank);
        }

        // For 2D case along dimension 1 (most common)
        if (rank == 2 && dim == 1) {
            return softmax2D();
        }

        throw new UnsupportedOperationException("Softmax for arbitrary dimensions not yet implemented");
    }

    /**
     * Softmax implementation for 2D tensors along dimension 1 (rows).
     */
    private Tensor softmax2D() {
        int rows = shape[0];
        int cols = shape[1];
        ComputeNode[] resultData = new ComputeNode[size];

        for (int i = 0; i < rows; i++) {
            // Find max for numerical stability
            ComputeNode maxVal = data[i * strides[0]];
            for (int j = 1; j < cols; j++) {
                ComputeNode val = data[i * strides[0] + j * strides[1]];
                maxVal = maxVal.max(val);
            }

            // Compute exp(x - max) and sum
            ComputeNode[] expVals = new ComputeNode[cols];
            ComputeNode sum = new ComputeNode(0.0);
            for (int j = 0; j < cols; j++) {
                ComputeNode val = data[i * strides[0] + j * strides[1]];
                expVals[j] = val.sub(maxVal).exp();
                sum = sum.add(expVals[j]);
            }

            // Normalize
            for (int j = 0; j < cols; j++) {
                resultData[i * cols + j] = expVals[j].div(sum);
            }
        }

        return new TensorImpl(resultData, shape, computeStrides(shape));
    }

    // ================================
    // Reduction Operations
    // ================================

    @Override
    public Tensor sum() {
        ComputeNode result = new ComputeNode(0.0);
        for (ComputeNode node : data) {
            result = result.add(node);
        }
        return new TensorImpl(new ComputeNode[]{result}, new int[]{1}, new int[]{1});
    }

    @Override
    public Tensor sum(int dim) {
        if (dim < 0 || dim >= rank) {
            throw new IllegalArgumentException("Dimension " + dim + " out of range for rank " + rank);
        }

        // For 2D case
        if (rank == 2) {
            return sum2D(dim);
        }

        throw new UnsupportedOperationException("Sum along dimension for arbitrary ranks not yet implemented");
    }

    private Tensor sum2D(int dim) {
        if (dim == 0) {
            // Sum along rows, return column vector
            ComputeNode[] resultData = new ComputeNode[shape[1]];
            for (int j = 0; j < shape[1]; j++) {
                ComputeNode sum = new ComputeNode(0.0);
                for (int i = 0; i < shape[0]; i++) {
                    sum = sum.add(data[i * strides[0] + j * strides[1]]);
                }
                resultData[j] = sum;
            }
            return new TensorImpl(resultData, new int[]{shape[1]}, new int[]{1});
        } else {
            // Sum along columns, return row vector
            ComputeNode[] resultData = new ComputeNode[shape[0]];
            for (int i = 0; i < shape[0]; i++) {
                ComputeNode sum = new ComputeNode(0.0);
                for (int j = 0; j < shape[1]; j++) {
                    sum = sum.add(data[i * strides[0] + j * strides[1]]);
                }
                resultData[i] = sum;
            }
            return new TensorImpl(resultData, new int[]{shape[0]}, new int[]{1});
        }
    }

    @Override
    public Tensor mean() {
        return sum().divide(createScalar(size));
    }

    @Override
    public Tensor mean(int dim) {
        Tensor sumTensor = sum(dim);
        return sumTensor.divide(createScalar(shape[dim]));
    }

    private Tensor createScalar(double value) {
        return new TensorImpl(new ComputeNode[]{new ComputeNode(value)}, new int[]{1}, new int[]{1});
    }

    // ================================
    // Utility Methods
    // ================================

    @Override
    public String toString() {
        if (size <= 20) {
            return toDetailedString();
        } else {
            return "Tensor(shape=" + Arrays.toString(shape) + ", size=" + size + ", rank=" + rank + ")";
        }
    }

    @Override
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape))
                .append(", size=").append(size)
                .append(", rank=").append(rank)
                .append(")\n");

        if (rank == 1) {
            sb.append("[");
            for (int i = 0; i < Math.min(size, 10); i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.3f", data[i].getValue()));
            }
            if (size > 10) sb.append(", ...");
            sb.append("]");
        } else if (rank == 2) {
            sb.append("[");
            for (int i = 0; i < Math.min(shape[0], 5); i++) {
                if (i > 0) sb.append(",\n ");
                sb.append("[");
                for (int j = 0; j < Math.min(shape[1], 5); j++) {
                    if (j > 0) sb.append(", ");
                    sb.append(String.format("%.3f", data[i * strides[0] + j * strides[1]].getValue()));
                }
                if (shape[1] > 5) sb.append(", ...");
                sb.append("]");
            }
            if (shape[0] > 5) sb.append(",\n ...");
            sb.append("]");
        } else {
            sb.append("Data preview: [");
            for (int i = 0; i < Math.min(size, 10); i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.3f", data[i].getValue()));
            }
            if (size > 10) sb.append(", ...");
            sb.append("]");
        }

        return sb.toString();
    }
}