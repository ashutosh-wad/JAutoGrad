package com.ashutoshwad.utils.jautograd.compute;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.DoubleSupplier;
import java.util.function.Function;

public class Matrix {
    private final ComputeNode[][] matrix;

    public Matrix(ComputeNode[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("Matrices with length 0 are not allowed!");
        }
        this.matrix = matrix;
    }

    public Matrix(Matrix m) {
        this.matrix = new ComputeNode[m.rowLength()][m.columnLength()];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                this.matrix[i][j] = m.matrix[i][j];
            }
        }
    }

    public static Matrix createXavierGlorotInitializedMatrix(int numRows, int numCols) {
        return createXavierGlorotInitializedMatrix(numRows, numCols, false);
    }

    public static Matrix createXavierGlorotInitializedMatrix(int numRows, int numCols, boolean trainable) {
        Random r = new Random();
        double scale = Math.sqrt(6.0/(numRows + numCols));
        return createMatrix(numRows, numCols, () -> ((r.nextDouble() * 2) - 1) * scale, trainable);
    }

    /**
     * Create a matrix of the specified dimensions where each value is initialized
     * to zero
     *
     * @param numRows the number of rows that the matrix should have
     * @param numCols the number of columns that the matrix should have
     * @return the newly created matrix
     */
    public static Matrix createMatrix(int numRows, int numCols) {
        return createMatrix(numRows, numCols, false);
    }

    public static Matrix createMatrix(int numRows, int numCols, boolean trainable) {
        return createMatrix(numRows, numCols, () -> 0.0, trainable);
    }

    /**
     * Create a matrix of the specified dimensions where each value is initialized
     * by the supplier specified
     *
     * @param numRows the number of rows that the matrix should have
     * @param numCols the number of columns that the matrix should have
     * @return the newly created matrix
     */
    public static Matrix createMatrix(int numRows, int numCols, DoubleSupplier supplier) {
        return createMatrix(numRows, numCols, supplier, false);
    }

    public static Matrix createMatrix(int numRows, int numCols, DoubleSupplier supplier, boolean trainable) {
        if (numRows == 0 || numCols == 0) {
            throw new IllegalArgumentException("Matrices with length 0 are not allowed!");
        }
        ComputeNode[][] retVal = new ComputeNode[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                retVal[i][j] = new ComputeNode(supplier.getAsDouble(), trainable);
            }
        }
        return new Matrix(retVal);
    }

    public int rowLength() {
        return matrix.length;
    }

    public int columnLength() {
        return matrix[0].length;
    }

    /**
     * Gets the compute node index at the specified row and column.
     *
     * @param row    The row with the element being queried.
     * @param column The column with the element being queried.
     * @return The node at the specified row and column
     */
    public ComputeNode get(int row, int column) {
        return matrix[row][column];
    }

    /**
     * Sets the compute node at the specified row and column.
     *
     * @param row    The row where the element is being replaced.
     * @param column The column where the element is being replaced.
     * @return The node that was at the specified row and column.
     */
    public ComputeNode set(int row, int column, ComputeNode node) {
        ComputeNode temp = matrix[row][column];
        matrix[row][column] = node;
        return temp;
    }

    /**
     * Multiplies this matrix with the provided matrix and returns the result.
     *
     * @param other The matrix by which this matrix is to be multiplied.
     * @return The result of the matrix multiplication.
     */
    public Matrix matmul(Matrix other) {
        int rowsA = matrix.length;
        int colsA = matrix[0].length;
        int rowsB = other.matrix.length;
        int colsB = other.matrix[0].length;

        if (colsA != rowsB) throw new IllegalArgumentException("Matrix dimensions don't match!");

        ComputeNode[][] result = new ComputeNode[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                ComputeNode sum = null;
                for (int k = 0; k < colsA; k++) {
                    if (null == sum) {
                        sum = matrix[i][k].mul(other.matrix[k][j]);
                    } else {
                        sum = sum.add(matrix[i][k].mul(other.matrix[k][j]));
                    }
                }
                result[i][j] = sum;
            }
        }

        return new Matrix(result);
    }

    /**
     * This method returns the transpose of the matrix.
     *
     * @return the transposed matrix
     */
    public Matrix transpose() {
        int rows = matrix.length;
        int cols = matrix[0].length;
        ComputeNode[][] transposed = new ComputeNode[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return new Matrix(transposed);
    }

    /**
     * This method performs the provided op for each value in this matrix and returns the result.
     *
     * @param operation the operation to be used to modify all the values in the
     *                  matrix
     * @return the modified matrix
     */
    public Matrix op(Function<ComputeNode, ComputeNode> operation) {
        ComputeNode[][] result = new ComputeNode[matrix.length][matrix[0].length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = operation.apply(matrix[i][j]);
            }
        }
        return new Matrix(result);
    }

    /**
     * Performs elementwise operations for the current matrix and the provided matrix. The required broadcasting operations are performed before the operation is applied.
     *
     * @param m         The other matrix
     * @param operation The operation to be performed.
     * @return
     */
    public Matrix elementwiseOp(Matrix m, BiFunction<ComputeNode, ComputeNode, ComputeNode> operation) {
        if (hasSameDimensions(this, m)) {
            ComputeNode[][] result = new ComputeNode[matrix.length][matrix[0].length];
            for (int i = 0; i < result.length; i++) {
                for (int j = 0; j < result[i].length; j++) {
                    result[i][j] = operation.apply(matrix[i][j], m.matrix[i][j]);
                }
            }
            return new Matrix(result);
        } else if (areDimenstionsBrodcastable(this, m)) {
            ComputeNode[][] result = new ComputeNode[Math.max(matrix.length, m.matrix.length)][Math.max(matrix[0].length, m.matrix[0].length)];
            for (int i = 0; i < result.length; i++) {
                for (int j = 0; j < result[i].length; j++) {
                    int i1 = (1 == matrix.length) ? 0 : i;
                    int i2 = (1 == m.matrix.length) ? 0 : i;
                    int j1 = (1 == matrix[0].length) ? 0 : j;
                    int j2 = (1 == m.matrix[0].length) ? 0 : j;
                    result[i][j] = operation.apply(matrix[i1][j1], m.matrix[i2][j2]);
                }
            }
            return new Matrix(result);
        } else {
            throw new IllegalArgumentException("The dimensions of the provided matrix are not compatible with this matrix for elementwise operations");
        }
    }

    private boolean areDimenstionsBrodcastable(Matrix m1, Matrix m2) {
        final int rowsA = m1.matrix.length;
        final int colsA = m1.matrix[0].length;
        final int rowsB = m2.matrix.length;
        final int colsB = m2.matrix[0].length;

        if (rowsA != rowsB && rowsA != 1 && rowsB != 1) {
            return false;
        }
        return colsA == colsB || colsA == 1 || colsB == 1;
    }

    private boolean hasSameDimensions(Matrix m1, Matrix m2) {
        return m1.matrix.length == m2.matrix.length && m1.matrix[0].length == m2.matrix[0].length;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sb.append(String.format("%9.5f", matrix[i][j].getValue()) + "\t");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                String value = String.format("%9.5f", matrix[i][j].getValue());
                String gradient = String.format("%9.5f", matrix[i][j].getGradient());
                sb.append("(" + value + ", " + gradient + ")" + "\t");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    //Start adding helper methods
    public Matrix add(Matrix m) {
        return elementwiseOp(m, (a, b) -> a.add(b));
    }

    public Matrix sub(Matrix m) {
        return elementwiseOp(m, (a, b) -> a.sub(b));
    }

    public Matrix div(Matrix m) {
        return elementwiseOp(m, (a, b) -> a.div(b));
    }

    public Matrix mul(Matrix m) {
        return elementwiseOp(m, (a, b) -> a.mul(b));
    }

    //Aggregator methods
    public Matrix sum() {
        Queue<ComputeNode> resultList = new LinkedList<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                resultList.add(matrix[i][j]);
            }
        }
        while(resultList.size() > 1) {
            ComputeNode a = resultList.poll();
            ComputeNode b = resultList.poll();
            resultList.add(a.add(b));
        }

        ComputeNode[][]retVal = new ComputeNode[1][1];
        retVal[0][0] = resultList.poll();
        return new Matrix(retVal);
    }

    //Adding compute node functions
    public Matrix sin() {
        return op(node -> node.sin());
    }
    public Matrix cos() {
        return op(node -> node.cos());
    }
    public Matrix tan() {
        return op(node -> node.tan());
    }
    public Matrix sinh() {
        return op(node -> node.sinh());
    }
    public Matrix cosh() {
        return op(node -> node.cosh());
    }
    public Matrix tanh() {
        return op(node -> node.tanh());
    }
    public Matrix relu() {
        return op(node -> node.relu());
    }
    public Matrix leakyRelu(double negSlope) {
        return op(node -> node.leakyRelu(negSlope));
    }
    public Matrix exp() {
        return op(node -> node.exp());
    }
    public Matrix ln() {
        return op(node -> node.ln());
    }
    public Matrix log() {
        return op(node -> node.log());
    }
    public Matrix sigmoid() {
        return op(node -> node.sigmoid());
    }
    public Matrix sqrt() {
        return op(node -> node.sqrt());
    }
    public Matrix simpleSwish() {
        return op(node -> node.simpleSwish());
    }
}
