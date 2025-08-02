package com.ashutoshwad.utils.jautograd.compute.tensor;

import com.ashutoshwad.utils.jautograd.compute.ComputeNode;

/**
 * A multi-dimensional tensor interface for storing and manipulating arrays of ComputeNodes.
 *
 * <p>This interface provides a high-level abstraction over multi-dimensional arrays, similar to
 * NumPy arrays or PyTorch tensors. Internally, the tensor data is stored as a single-dimensional
 * array with stride-based indexing for efficient memory access and manipulation.</p>
 *
 * <p><strong>Key Concepts:</strong></p>
 * <ul>
 *   <li><strong>Shape:</strong> The dimensions of the tensor (e.g., [2, 3, 4] for a 3D tensor)</li>
 *   <li><strong>Strides:</strong> Number of elements to skip in each dimension when traversing</li>
 *   <li><strong>Rank:</strong> Number of dimensions (e.g., matrix has rank 2)</li>
 * </ul>
 *
 * <p><strong>Usage Examples:</strong></p>
 * <pre>{@code
 * // Create a 2x3 matrix
 * Tensor matrix = Tensor.zeros(2, 3);
 * matrix.set(new ComputeNode(5.0), 0, 1);  // Set element at row 0, column 1
 *
 * // Create a 3D tensor (batch_size=2, height=4, width=5)
 * Tensor tensor3d = Tensor.random(2, 4, 5);
 * ComputeNode value = tensor3d.get(1, 2, 3);  // Get element at [1][2][3]
 *
 * // Matrix operations
 * Tensor a = Tensor.ones(3, 4);
 * Tensor b = Tensor.ones(4, 2);
 * Tensor result = a.matmul(b);  // 3x2 result
 * }</pre>
 *
 * @author Ashutosh Wad
 */
public interface Tensor {

    // ================================
    // Core Properties and Access
    // ================================

    /**
     * Returns the shape of this tensor as an array of dimensions.
     *
     * <p>For example, a 2x3 matrix returns [2, 3], and a 3D tensor of size 2x4x5 returns [2, 4, 5].</p>
     *
     * @return array containing the size of each dimension
     */
    int[] getShape();

    /**
     * Returns the number of dimensions (rank) of this tensor.
     *
     * <p>Examples:</p>
     * <ul>
     *   <li>Scalar: rank 0</li>
     *   <li>Vector: rank 1</li>
     *   <li>Matrix: rank 2</li>
     *   <li>3D tensor: rank 3</li>
     * </ul>
     *
     * @return number of dimensions
     */
    int getRank();

    /**
     * Returns the total number of elements in this tensor.
     *
     * <p>This is the product of all dimensions. For example, a 2x3x4 tensor has 24 elements.</p>
     *
     * @return total number of elements
     */
    int getSize();

    /**
     * Gets the ComputeNode at the specified multi-dimensional index.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor matrix = Tensor.zeros(3, 4);
     * ComputeNode value = matrix.get(1, 2);  // Get element at row 1, column 2
     *
     * Tensor tensor3d = Tensor.zeros(2, 3, 4);
     * ComputeNode value = tensor3d.get(0, 1, 2);  // Get element at [0][1][2]
     * }</pre>
     *
     * @param indices the multi-dimensional index (must match tensor rank)
     * @return the ComputeNode at the specified position
     * @throws IllegalArgumentException if number of indices doesn't match tensor rank
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    ComputeNode get(int... indices);

    /**
     * Sets the ComputeNode at the specified multi-dimensional index.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor matrix = Tensor.zeros(3, 4);
     * matrix.set(new ComputeNode(5.0), 1, 2);  // Set element at row 1, column 2
     *
     * Tensor tensor3d = Tensor.zeros(2, 3, 4);
     * tensor3d.set(new ComputeNode(7.5), 0, 1, 2);  // Set element at [0][1][2]
     * }</pre>
     *
     * @param value the ComputeNode to store
     * @param indices the multi-dimensional index (must match tensor rank)
     * @throws IllegalArgumentException if number of indices doesn't match tensor rank
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    void set(ComputeNode value, int... indices);

    // ================================
    // Factory Methods
    // ================================

    /**
     * Creates a tensor filled with zeros.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor vector = Tensor.zeros(5);        // 1D vector of length 5
     * Tensor matrix = Tensor.zeros(3, 4);     // 3x4 matrix
     * Tensor tensor3d = Tensor.zeros(2, 3, 4); // 2x3x4 tensor
     * }</pre>
     *
     * @param shape the dimensions of the tensor
     * @return a new tensor filled with ComputeNode(0.0)
     * @throws IllegalArgumentException if any dimension is <= 0
     */
    static Tensor zeros(int... shape) {
        // Implementation would be provided by concrete class
        throw new UnsupportedOperationException("Must be implemented by concrete class");
    }

    /**
     * Creates a tensor filled with ones.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor vector = Tensor.ones(5);        // [1, 1, 1, 1, 1]
     * Tensor matrix = Tensor.ones(2, 3);     // 2x3 matrix of ones
     * }</pre>
     *
     * @param shape the dimensions of the tensor
     * @return a new tensor filled with ComputeNode(1.0)
     * @throws IllegalArgumentException if any dimension is <= 0
     */
    static Tensor ones(int... shape) {
        throw new UnsupportedOperationException("Must be implemented by concrete class");
    }

    /**
     * Creates a tensor filled with random values from a standard normal distribution (mean=0, std=1).
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor weights = Tensor.random(256, 64);  // Random weight matrix for neural network
     * Tensor bias = Tensor.random(64);          // Random bias vector
     * }</pre>
     *
     * @param shape the dimensions of the tensor
     * @return a new tensor filled with random ComputeNodes
     * @throws IllegalArgumentException if any dimension is <= 0
     */
    static Tensor random(int... shape) {
        throw new UnsupportedOperationException("Must be implemented by concrete class");
    }

    /**
     * Creates a tensor from existing ComputeNode data.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * ComputeNode[][] matrix = {{new ComputeNode(1), new ComputeNode(2)},
     *                           {new ComputeNode(3), new ComputeNode(4)}};
     * Tensor tensor = Tensor.from(matrix);  // Creates 2x2 tensor
     * }</pre>
     *
     * @param data multi-dimensional array of ComputeNodes
     * @return a new tensor containing the provided data
     * @throws IllegalArgumentException if data is null or has inconsistent dimensions
     */
    static Tensor from(ComputeNode[]... data) {
        throw new UnsupportedOperationException("Must be implemented by concrete class");
    }

    // ================================
    // Shape Operations
    // ================================

    /**
     * Returns a new tensor with the specified shape, sharing the same underlying data.
     *
     * <p>The new shape must have the same total number of elements as the original tensor.
     * This operation is very efficient as it only changes the stride information.</p>
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor vector = Tensor.ones(12);           // Shape: [12]
     * Tensor matrix = vector.reshape(3, 4);      // Shape: [3, 4] - same data
     * Tensor tensor3d = vector.reshape(2, 2, 3); // Shape: [2, 2, 3] - same data
     * }</pre>
     *
     * @param newShape the desired shape
     * @return a new tensor view with the specified shape
     * @throws IllegalArgumentException if the new shape has a different total size
     */
    Tensor reshape(int... newShape);

    /**
     * Returns a new tensor that is the transpose of this tensor.
     *
     * <p>For 2D tensors (matrices), this swaps rows and columns.
     * For higher-dimensional tensors, this reverses all dimensions.</p>
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor matrix = Tensor.ones(3, 4);      // 3x4 matrix
     * Tensor transposed = matrix.transpose(); // 4x3 matrix
     * }</pre>
     *
     * @return a new tensor that is the transpose of this tensor
     */
    Tensor transpose();

    /**
     * Returns a tensor with dimensions swapped according to the specified axis order.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor tensor = Tensor.ones(2, 3, 4);        // Shape: [2, 3, 4]
     * Tensor permuted = tensor.transpose(2, 0, 1); // Shape: [4, 2, 3]
     * // Dimension 0 -> position 1, dimension 1 -> position 2, dimension 2 -> position 0
     * }</pre>
     *
     * @param axes the new order of dimensions
     * @return a new tensor with dimensions reordered
     * @throws IllegalArgumentException if axes length doesn't match rank or contains invalid indices
     */
    Tensor transpose(int... axes);

    // ================================
    // Mathematical Operations
    // ================================

    /**
     * Element-wise addition of this tensor with another tensor.
     *
     * <p>Both tensors must have the same shape, or one must be broadcastable to the other.</p>
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor a = Tensor.ones(2, 3);
     * Tensor b = Tensor.ones(2, 3);
     * Tensor result = a.add(b);  // Element-wise addition
     * }</pre>
     *
     * @param other the tensor to add
     * @return a new tensor containing the element-wise sum
     * @throws IllegalArgumentException if tensors are not compatible for addition
     */
    Tensor add(Tensor other);

    /**
     * Element-wise subtraction of another tensor from this tensor.
     *
     * @param other the tensor to subtract
     * @return a new tensor containing the element-wise difference
     * @throws IllegalArgumentException if tensors are not compatible for subtraction
     */
    Tensor subtract(Tensor other);

    /**
     * Element-wise multiplication of this tensor with another tensor.
     *
     * <p><strong>Note:</strong> This is element-wise multiplication, not matrix multiplication.
     * For matrix multiplication, use {@link #matmul(Tensor)}.</p>
     *
     * @param other the tensor to multiply element-wise
     * @return a new tensor containing the element-wise product
     * @throws IllegalArgumentException if tensors are not compatible for element-wise multiplication
     */
    Tensor multiply(Tensor other);

    /**
     * Element-wise division of this tensor by another tensor.
     *
     * @param other the tensor to divide by
     * @return a new tensor containing the element-wise quotient
     * @throws IllegalArgumentException if tensors are not compatible for division
     */
    Tensor divide(Tensor other);

    /**
     * Matrix multiplication of this tensor with another tensor.
     *
     * <p>For 2D tensors: standard matrix multiplication.
     * For higher-dimensional tensors: batched matrix multiplication on the last two dimensions.</p>
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor a = Tensor.random(3, 4);    // 3x4 matrix
     * Tensor b = Tensor.random(4, 2);    // 4x2 matrix
     * Tensor result = a.matmul(b);       // 3x2 result
     *
     * // Batched matrix multiplication
     * Tensor batch_a = Tensor.random(5, 3, 4);  // Batch of 5 3x4 matrices
     * Tensor batch_b = Tensor.random(5, 4, 2);  // Batch of 5 4x2 matrices
     * Tensor batch_result = batch_a.matmul(batch_b);  // Batch of 5 3x2 matrices
     * }</pre>
     *
     * @param other the tensor to multiply with
     * @return a new tensor containing the matrix multiplication result
     * @throws IllegalArgumentException if tensor dimensions are incompatible for matrix multiplication
     */
    Tensor matmul(Tensor other);

    // ================================
    // Activation Functions
    // ================================

    /**
     * Applies the ReLU activation function element-wise.
     *
     * <p>ReLU(x) = max(0, x)</p>
     *
     * @return a new tensor with ReLU applied to each element
     */
    Tensor relu();

    /**
     * Applies the Leaky ReLU activation function element-wise.
     *
     * <p>LeakyReLU(x) = x if x > 0, else negativeSlope * x</p>
     *
     * @param negativeSlope the slope for negative values (typically 0.01)
     * @return a new tensor with Leaky ReLU applied to each element
     */
    Tensor leakyRelu(double negativeSlope);

    /**
     * Applies the softmax function along the specified dimension.
     *
     * <p>Softmax converts logits to probabilities that sum to 1 along the specified axis.</p>
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor logits = Tensor.random(3, 4);        // 3 samples, 4 classes
     * Tensor probabilities = logits.softmax(1);   // Softmax along classes (dim 1)
     * // Each row now sums to 1.0
     * }</pre>
     *
     * @param dim the dimension along which to apply softmax
     * @return a new tensor with softmax applied along the specified dimension
     * @throws IllegalArgumentException if dim is out of bounds
     */
    Tensor softmax(int dim);

    // ================================
    // Reduction Operations
    // ================================

    /**
     * Computes the sum of all elements in the tensor.
     *
     * @return a scalar tensor containing the sum
     */
    Tensor sum();

    /**
     * Computes the sum along the specified dimension.
     *
     * <p><strong>Usage:</strong></p>
     * <pre>{@code
     * Tensor matrix = Tensor.ones(3, 4);    // 3x4 matrix of ones
     * Tensor rowSums = matrix.sum(1);       // Sum along columns, result shape: [3]
     * Tensor colSums = matrix.sum(0);       // Sum along rows, result shape: [4]
     * }</pre>
     *
     * @param dim the dimension along which to sum
     * @return a new tensor with one fewer dimension
     * @throws IllegalArgumentException if dim is out of bounds
     */
    Tensor sum(int dim);

    /**
     * Computes the mean of all elements in the tensor.
     *
     * @return a scalar tensor containing the mean
     */
    Tensor mean();

    /**
     * Computes the mean along the specified dimension.
     *
     * @param dim the dimension along which to compute the mean
     * @return a new tensor with one fewer dimension
     * @throws IllegalArgumentException if dim is out of bounds
     */
    Tensor mean(int dim);

    // ================================
    // Utility Methods
    // ================================

    /**
     * Returns a string representation of this tensor.
     *
     * <p>For small tensors, this shows all elements. For large tensors, this shows
     * a summary with shape information and sample elements.</p>
     *
     * @return string representation of the tensor
     */
    @Override
    String toString();

    /**
     * Returns a detailed string representation showing shape, size, and data type information.
     *
     * <p><strong>Example output:</strong></p>
     * <pre>
     * Tensor(shape=[2, 3], size=6, rank=2)
     * [[1.0, 2.0, 3.0],
     *  [4.0, 5.0, 6.0]]
     * </pre>
     *
     * @return detailed string representation
     */
    String toDetailedString();
}