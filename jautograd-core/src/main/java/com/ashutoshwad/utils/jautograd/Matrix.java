package com.ashutoshwad.utils.jautograd;

import java.util.Random;

public class Matrix extends ComputeNode {
    public Matrix(int rows, int cols) {
        super(new MatrixStore(rows, cols), false);
    }

    public Matrix(int rows, int cols, boolean requiresGradient) {
        super(new MatrixStore(rows, cols), requiresGradient);
    }

    protected Matrix(MatrixStore values, ComputeNode...parents) {
        super(values, parents);
    }

    public static Matrix createXavierGlorotInitializedMatrix(int rows, int columns) {
        return createXavierGlorotInitializedMatrix(rows, columns, false);
    }

    public static Matrix createXavierGlorotInitializedMatrix(int rows, int columns, boolean requiresGradient) {
        Random r = new Random();
        final double scale = (float)Math.sqrt(6.0 / (rows + columns));
        Matrix m = new Matrix(rows, columns, requiresGradient);
        MatrixStore values = m.getValues();
        float[]backingArray = values.backingArray();
        for (int i = 0; i < backingArray.length; i++) {
            backingArray[i] = (float)((r.nextDouble()*2-1)*scale);
        }
        return m;
    }

    /* Start unary operators */
    public Matrix sqrt() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.SQRT, FunctionRegistry.SQRT_GRAD, "SQRT");
    }
    public Matrix sin() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.SIN, FunctionRegistry.SIN_GRAD, "SIN");
    }
    public Matrix cos() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.COS, FunctionRegistry.COS_GRAD, "COS");
    }
    public Matrix tan() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.TAN, FunctionRegistry.TAN_GRAD, "TAN");
    }
    public Matrix sinh() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.SINH, FunctionRegistry.SINH_GRAD, "SINH");
    }
    public Matrix cosh() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.COSH, FunctionRegistry.COSH_GRAD, "COSH");
    }
    public Matrix tanh() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.TANH, FunctionRegistry.TANH_GRAD, "TANH");
    }
    public Matrix relu() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.RELU, FunctionRegistry.RELU_GRAD, "RELU");
    }
    public Matrix reluLeaky() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.LEAKY_RELU, FunctionRegistry.LEAKY_RELU_GRAD, "LEAKY_RELU");
    }
    public Matrix exp() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.EXP, FunctionRegistry.EXP_GRAD, "EXP");
    }
    public Matrix ln() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.LN, FunctionRegistry.LN_GRAD, "LN");
    }
    public Matrix log() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.LOG, FunctionRegistry.LOG_GRAD, "LOG");
    }
    public Matrix swish() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.SIMPLE_SWISH, FunctionRegistry.SIMPLE_SWISH_GRAD, "SWISH");
    }
    public Matrix sigmoid() {
        return new ElementwiseUnaryOperator(this, FunctionRegistry.SIGMOID, FunctionRegistry.SIGMOID_GRAD, "SIGMOID");
    }

    /* Start binary operators */
    public Matrix add(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.ADD, FunctionRegistry.ADD_GRAD_LEFT, FunctionRegistry.ADD_GRAD_RIGHT, "ADD");
    }
    public Matrix sub(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.SUB, FunctionRegistry.SUB_GRAD_LEFT, FunctionRegistry.SUB_GRAD_RIGHT, "SUB");
    }
    public Matrix mul(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.MUL, FunctionRegistry.MUL_GRAD_LEFT, FunctionRegistry.MUL_GRAD_RIGHT, "MUL");
    }
    public Matrix div(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.DIV, FunctionRegistry.DIV_GRAD_LEFT, FunctionRegistry.DIV_GRAD_RIGHT, "DIV");
    }
    public Matrix pow(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.POW, FunctionRegistry.POW_GRAD_LEFT, FunctionRegistry.POW_GRAD_RIGHT, "POW");
    }
    public Matrix max(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.MAX, FunctionRegistry.MAX_GRAD_LEFT, FunctionRegistry.MAX_GRAD_RIGHT, "MAX");
    }
    public Matrix min(ComputeNode other) {
        return new ElementwiseBinaryOperator(this, other, FunctionRegistry.MIN, FunctionRegistry.MIN_GRAD_LEFT, FunctionRegistry.MIN_GRAD_RIGHT, "MIN");
    }
    public Matrix matmul(ComputeNode other) {
        return new MultiplicationOperator(this, other, "MATMUL");
    }

    /* Override mandatory methods */
    @Override
    public void computeResult() {
        //Do nothing as this is a leaf node
    }

    @Override
    public void backpropogateGradients() {
        //Do nothing as this is a leaf node
    }
}
