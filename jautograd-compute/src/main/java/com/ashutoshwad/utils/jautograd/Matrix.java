package com.ashutoshwad.utils.jautograd;

import java.util.Random;
import java.util.function.Supplier;

public class Matrix extends AbstractMatrix {
    public static final double EPSILON = 0.00000001;
    private static final Random random = new Random();

    private record OperationTuple(Matrix a,
                                  Matrix b,
                                  double[][] value,
                                  double[][] gradient,
                                  boolean requiresGradient,
                                  ForwardComputeOperation forwardComputeOperation,
                                  BackwardComputeOperation backwardComputeOperation) {}

    private OperationTuple commonElementwiseBinaryOperationSteps(Matrix other, FunctionRegistry.BinaryCalcFunction calcFunction, FunctionRegistry.BinaryGradientFunction lGrad, FunctionRegistry.BinaryGradientFunction rGrad) {
        int numRows = Math.max(numRows(), other.numRows());
        int numCols = Math.max(numCols(), other.numCols());
        Matrix a = BroadcastView.broadcast(this, numRows, numCols);
        Matrix b = BroadcastView.broadcast(other, numRows, numCols);
        double[][] value = new double[numRows][numCols];
        double[][] gradient = null;
        boolean requiresGradient = this.requiresGradient || other.requiresGradient;
        if (requiresGradient) {
            gradient = new double[numRows][numCols];
        }
        ForwardComputeOperation forwardComputeOperation = new ElementWiseBinaryForwardOperation(a, b, calcFunction, a.forwardComputeOperation, b.forwardComputeOperation);
        BackwardComputeOperation backwardComputeOperation = null;
        if (requiresGradient) {
            backwardComputeOperation = new ElementWiseBinaryBackwardOperation(a, b, lGrad, rGrad, a.backwardComputeOperation, b.backwardComputeOperation);
        }
        return new OperationTuple(a, b, value, gradient, requiresGradient, forwardComputeOperation, backwardComputeOperation);
    }

    private OperationTuple commonElementwiseUnaryOperationSteps(FunctionRegistry.UnaryCalcFunction calcFunction, FunctionRegistry.UnaryGradientFunction gradFunc) {
        double[][] value = new double[this.numRows()][this.numCols()];
        double[][] gradient = null;
        if (requiresGradient) {
            gradient = new double[this.numRows()][this.numCols()];
        }
        ForwardComputeOperation forwardComputeOperation = new ElementWiseUnaryForwardOperation(this, calcFunction, this.forwardComputeOperation);
        BackwardComputeOperation backwardComputeOperation = null;
        if (requiresGradient) {
            backwardComputeOperation = new ElementWiseUnaryBackwardOperation(this, gradFunc, this.backwardComputeOperation);
        }
        return new OperationTuple(this, null, value, gradient, requiresGradient, forwardComputeOperation, backwardComputeOperation);
    }

    //Elementwise binary operations start here
    public Matrix add(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.ADD, FunctionRegistry.ADD_GRAD_LEFT, FunctionRegistry.ADD_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix sub(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.SUB, FunctionRegistry.SUB_GRAD_LEFT, FunctionRegistry.SUB_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix div(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.DIV, FunctionRegistry.DIV_GRAD_LEFT, FunctionRegistry.DIV_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix mul(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.MUL, FunctionRegistry.MUL_GRAD_LEFT, FunctionRegistry.MUL_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix pow(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.POW, FunctionRegistry.POW_GRAD_LEFT, FunctionRegistry.POW_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix max(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.MAX, FunctionRegistry.MAX_GRAD_LEFT, FunctionRegistry.MAX_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix min(Matrix other) {
        OperationTuple tup = commonElementwiseBinaryOperationSteps(other, FunctionRegistry.MIN, FunctionRegistry.MIN_GRAD_LEFT, FunctionRegistry.MIN_GRAD_RIGHT);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix matmul(Matrix b) {
        if(numCols() != b.numRows()) {
            String errorMessage = "Matrix dimension mismatch, this(rows: %d, columns: %d) and otherthis(rows: %d, columns: %d), cannot multiply!";
            throw new IllegalArgumentException(String.format(errorMessage, numRows(), numCols(), b.numRows(), b.numCols()));
        }
        int numRows = numRows();
        int numCols = b.numCols();
        Matrix a = this;
        double[][] value = new double[numRows][numCols];
        double[][] gradient = null;
        boolean requiresGradient = this.requiresGradient || b.requiresGradient;
        if (requiresGradient) {
            gradient = new double[numRows][numCols];
        }
        ForwardComputeOperation forwardComputeOperation = new MatrixMultiplicationForwardOperation(a, b, a.forwardComputeOperation, b.forwardComputeOperation);
        BackwardComputeOperation backwardComputeOperation = null;
        if (requiresGradient) {
            backwardComputeOperation = new MatrixMultiplicationBackwardOperation(a, b, a.backwardComputeOperation, b.backwardComputeOperation);
        }
        return new Matrix(value, gradient, requiresGradient, forwardComputeOperation, backwardComputeOperation);
    }
    //Elementwise unary operations start here
    public Matrix sqrt() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.SQRT, FunctionRegistry.SQRT_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix sin() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.SIN, FunctionRegistry.SIN_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix cos() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.COS, FunctionRegistry.COS_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix tan() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.TAN, FunctionRegistry.TAN_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix sinh() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.SINH, FunctionRegistry.SINH_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix cosh() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.COSH, FunctionRegistry.COSH_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix tanh() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.TANH, FunctionRegistry.TANH_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix relu() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.RELU, FunctionRegistry.RELU_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix leakyrelu() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.LEAKY_RELU, FunctionRegistry.LEAKY_RELU_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix exp() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.EXP, FunctionRegistry.EXP_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix ln() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.LN, FunctionRegistry.LN_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix log() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.LOG, FunctionRegistry.LOG_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix sigmoid() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.SIGMOID, FunctionRegistry.SIGMOID_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }
    public Matrix swish() {
        OperationTuple tup = commonElementwiseUnaryOperationSteps(FunctionRegistry.SIMPLE_SWISH, FunctionRegistry.SIMPLE_SWISH_GRAD);
        return new Matrix(tup.value(), tup.gradient(), tup.requiresGradient, tup.forwardComputeOperation, tup.backwardComputeOperation);
    }

    /*Matrix reduction methods start here*/
    public Matrix sum() {
        double[][] value = new double[1][1];
        ForwardComputeOperation fop = new SumForwardOperation(this, this.forwardComputeOperation);

        double[][] gradient = null;
        BackwardComputeOperation bop = null;
        if (requiresGradient) {
            gradient = new double[1][1];
            bop = new SumBackwardOperation(this, this.backwardComputeOperation);
        }

        return new Matrix(value, gradient, requiresGradient, fop, bop);
    }

    /**
     * Sum across a given axis. 0 indicates sum across rows, so result will have 1 row, 1 indicates sum across columns so result will have 1 column.
     * @param axis 0 indicates sum across rows, so result will have 1 row, 1 indicates sum across columns so result will have 1 column.
     * @return The resultant matrix
     */
    public Matrix sum(final int axis) {
        if (axis == 0 && this.numRows() == 1) {
            return this;
        }
        if (axis == 1 && this.numCols() == 1){
            return this;
        }
        if (axis != 0 && axis !=1) {
            throw new IllegalArgumentException("Axis must be either 0 for rows or 1 for columns. Found: " + axis);
        }

        if(axis == 0) {
            double[][] value = new double[1][this.numCols()];
            ForwardComputeOperation fop = new SumForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[1][this.numCols()];
                bop = new SumBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        } else {
            double[][] value = new double[this.numRows()][1];
            ForwardComputeOperation fop = new SumForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[this.numRows()][1];
                bop = new SumBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        }
    }

    public Matrix max() {
        double[][] value = new double[1][1];
        ForwardComputeOperation fop = new MaxForwardOperation(this, this.forwardComputeOperation);

        double[][] gradient = null;
        BackwardComputeOperation bop = null;
        if (requiresGradient) {
            gradient = new double[1][1];
            bop = new MaxBackwardOperation(this, this.backwardComputeOperation);
        }

        return new Matrix(value, gradient, requiresGradient, fop, bop);
    }

    /**
     * Find max across a given axis. 0 indicates max across rows, so result will have 1 row, 1 indicates max across columns so result will have 1 column.
     * @param axis 0 indicates max across rows, so result will have 1 row, 1 indicates max across columns so result will have 1 column.
     * @return The resultant matrix
     */
    public Matrix max(final int axis) {
        if (axis == 0 && this.numRows() == 1) {
            return this;
        }
        if (axis == 1 && this.numCols() == 1){
            return this;
        }
        if (axis != 0 && axis !=1) {
            throw new IllegalArgumentException("Axis must be either 0 for rows or 1 for columns. Found: " + axis);
        }

        if(axis == 0) {
            double[][] value = new double[1][this.numCols()];
            ForwardComputeOperation fop = new MaxForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[1][this.numCols()];
                bop = new MaxBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        } else {
            double[][] value = new double[this.numRows()][1];
            ForwardComputeOperation fop = new MaxForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[this.numRows()][1];
                bop = new MaxBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        }
    }

    public Matrix min() {
        double[][] value = new double[1][1];
        ForwardComputeOperation fop = new MinForwardOperation(this, this.forwardComputeOperation);

        double[][] gradient = null;
        BackwardComputeOperation bop = null;
        if (requiresGradient) {
            gradient = new double[1][1];
            bop = new MinBackwardOperation(this, this.backwardComputeOperation);
        }

        return new Matrix(value, gradient, requiresGradient, fop, bop);
    }

    /**
     * Find min across a given axis. 0 indicates min across rows, so result will have 1 row, 1 indicates min across columns so result will have 1 column.
     * @param axis 0 indicates min across rows, so result will have 1 row, 1 indicates min across columns so result will have 1 column.
     * @return The resultant matrix
     */
    public Matrix min(final int axis) {
        if (axis == 0 && this.numRows() == 1) {
            return this;
        }
        if (axis == 1 && this.numCols() == 1){
            return this;
        }
        if (axis != 0 && axis !=1) {
            throw new IllegalArgumentException("Axis must be either 0 for rows or 1 for columns. Found: " + axis);
        }

        if(axis == 0) {
            double[][] value = new double[1][this.numCols()];
            ForwardComputeOperation fop = new MinForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[1][this.numCols()];
                bop = new MinBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        } else {
            double[][] value = new double[this.numRows()][1];
            ForwardComputeOperation fop = new MinForwardByAxisOperation(this, axis, this.forwardComputeOperation);

            double[][] gradient = null;
            BackwardComputeOperation bop = null;

            if (requiresGradient) {
                gradient = new double[this.numRows()][1];
                bop = new MinBackwardByAxisOperation(this, axis, this.backwardComputeOperation);
            }

            return new Matrix(value, gradient, requiresGradient, fop, bop);
        }
    }

    public Matrix mean() {
        return sum().div(create((double)numRows() * numCols()));
    }

    public Matrix mean(int axis) {
        return sum(axis).div(create((axis == 0) ? (double)numRows() : (double)numCols()));
    }

    public Matrix variance() {
        Matrix intermediate = sub(mean());
        return intermediate.mul(intermediate).mean();
    }

    public Matrix variance(int axis) {
        Matrix intermediate = sub(mean(axis));
        return intermediate.mul(intermediate).mean(axis);
    }

    public Matrix std() {
        return variance().sqrt();
    }

    public Matrix std(int axis) {
        return variance(axis).sqrt();
    }

    public Matrix softmax() {
        Matrix maxVal = max();
        Matrix shifted = sub(maxVal);
        Matrix shiftedExp = shifted.exp();
        Matrix sum = shiftedExp.sum();
        return shiftedExp.div(sum.add(create(EPSILON)));
    }

    public Matrix softmax(int axis) {
        Matrix maxVal = max(axis);
        Matrix shifted = sub(maxVal);
        Matrix shiftedExp = shifted.exp();
        Matrix sum = shiftedExp.sum(axis);
        return shiftedExp.div(sum.add(create(EPSILON)));
    }

    public Matrix layerNorm() {
        return sub(mean()).div(variance().add(create(EPSILON)).sqrt());
    }

    public Matrix layerNorm(int axis) {
        return sub(mean(axis)).div(variance(axis).add(create(EPSILON)).sqrt());
    }

    /* Matrix training dropout methods start here */
    public Matrix dropout() {
        return dropout(0.5);
    }

    public Matrix dropout(double rate) {
        return dropout(()->rate);
    }

    public Matrix dropout(Supplier<Double>rate) {
        return new DropoutView(this, rate);
    }

    /*Matrix split methods start here*/
    public Matrix[] splitRows(int numParts) {
        int rowsPerPart = numRows() / numParts;
        Matrix[] parts = new Matrix[numParts];

        for (int i = 0; i < numParts; i++) {
            int start = i * rowsPerPart;
            int length = (i == numParts - 1) ? numRows() - start : rowsPerPart; // Handle remainder
            parts[i] = new SplitView(this, start, length, 0, numCols());
        }
        return parts;
    }

    public Matrix[] splitCols(int numParts) {
        int colsPerPart = numCols() / numParts;
        Matrix[] parts = new Matrix[numParts];

        for (int i = 0; i < numParts; i++) {
            int start = i * colsPerPart;
            int length = (i == numParts - 1) ? numCols() - start : colsPerPart;
            parts[i] = new SplitView(this, 0, numRows(), start, length);
        }
        return parts;
    }

    // Get arbitrary slice
    public Matrix slice(int rowStart, int numRows, int colStart, int numCols) {
        return new SplitView(this, rowStart, numRows, colStart, numCols);
    }

    // Get single row/column
    public Matrix row(int index) {
        return new SplitView(this, index, 1, 0, numCols());
    }

    public Matrix col(int index) {
        return new SplitView(this, 0, numRows(), index, 1);
    }

    public Matrix[][] explode() {
        Matrix[][]blownUp = new Matrix[numRows()][numCols()];
        for (int i = 0; i < blownUp.length; i++) {
            for (int j = 0; j < blownUp[i].length; j++) {
                blownUp[i][j] = new SplitView(this, i, 1, j, 1);
            }
        }
        return blownUp;
    }

    public Matrix transpose() {
        return new TransposeView(this);
    }

    /*Matrix concat methods start here*/
    public Matrix concat(Matrix...other){
        return concatHorizontal(other);
    }
    public Matrix concatHorizontal(Matrix...other) {
        Matrix[]mats = new Matrix[other.length + 1];
        mats[0] = this;
        for (int i = 1; i < mats.length; i++) {
            mats[i] = other[i-1];
        }
        return new HorizontalConcatView(mats);
    }
    public Matrix concatVertical(Matrix...other){
        Matrix[]mats = new Matrix[other.length + 1];
        mats[0] = this;
        for (int i = 1; i < mats.length; i++) {
            mats[i] = other[i-1];
        }
        return new VerticalConcatView(mats);
    }

    /*Matrix forward and backward methods*/
    public void forward() {
        if(null!=forwardComputeOperation){
            forwardComputeOperation.forward();
        }
    }

    public void zeroGradAndforward() {
        if(null!=forwardComputeOperation) {
            forwardComputeOperation.zeroGradAndforward();
        }
    }

    public void zeroGrad() {
        if(null!=forwardComputeOperation) {
            forwardComputeOperation.zeroGrad();
        }
    }

    public void backward() {
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numCols(); col++) {
                setGradient(row, col, 1);
            }
        }
        if(null!=backwardComputeOperation) {
            backwardComputeOperation.backward();
        }
    }
    /*Matrix specific static methods start*/

    protected Matrix(double[][] value, double[][] gradient, boolean requiresGradient, ForwardComputeOperation forwardComputeOperation, BackwardComputeOperation backwardComputeOperation) {
        super(value, gradient, requiresGradient, forwardComputeOperation, backwardComputeOperation);
        if(null!=forwardComputeOperation) {
            forwardComputeOperation.setResult(this);
            forwardComputeOperation.perform();
        }
        if(null!=backwardComputeOperation) {
            backwardComputeOperation.setResult(this);
        }
    }

    /**
     * This is a constructor used by Views. As views are passthrough and do not have internal states,
     * it ensures that the result matrix is never overwritten in either the forward or backward operations.
     */
    protected Matrix(boolean requiresGradient, ForwardComputeOperation forwardComputeOperation, BackwardComputeOperation backwardComputeOperation) {
        super(null, null, requiresGradient, forwardComputeOperation, backwardComputeOperation);
    }

    public static Matrix create(double val) {
        return create(val, false);
    }

    public static Matrix create(double val, boolean trainable) {
        return create(1, 1, () -> val, trainable);
    }

    public static Matrix create(int rows, int columns) {
        return create(rows, columns, () -> 0.0);
    }

    public static Matrix create(int rows, int columns, boolean trainable) {
        return create(rows, columns, () -> 0.0, trainable);
    }

    public static Matrix create(int rows, int columns, Supplier<Double> valueSupplier) {
        return create(rows, columns, valueSupplier, false);
    }

    public static Matrix create(int rows, int columns, Supplier<Double> valueSupplier, boolean trainable) {
        if (rows <= 0) {
            throw new IllegalArgumentException("A matrix cannot have 0 or less than 0 rows.");
        } if (columns <= 0) {
            throw new IllegalArgumentException("A matrix cannot have 0 or less than 0 columns.");
        } double[][] value = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value[i][j] = valueSupplier.get();
            }
        }
        double[][] gradient = null;
        if (trainable) {
            gradient = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    gradient[i][j] = 0;
                }
            }
        }
        return new Matrix(value, gradient, trainable, null, null);
    }

    public static Matrix createXavierGlorotInitializedMatrix(int rows, int columns, boolean trainable) {
        final double scale = Math.sqrt(6.0 / (rows + columns));
        return create(rows, columns, () -> ((random.nextDouble() * 2 - 1) * scale), trainable);
    }

    /*Matrix specific static methods end*/
}
