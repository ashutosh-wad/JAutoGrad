package com.ashutoshwad.utils.jautograd;

public class ElementwiseBinaryOperator extends Matrix {
    private final FunctionRegistry.BinaryCalcFunction forwardFunction;
    private final FunctionRegistry.BinaryGradientFunction leftGradientFunction;
    private final FunctionRegistry.BinaryGradientFunction rightGradientFunction;
    private final ComputeNode left;
    private final ComputeNode right;
    public final String operatorName;

    private boolean broadcastLeftRows;
    private boolean broadcastLeftColumns;

    private boolean broadcastRightRows;
    private boolean broadcastRightColumns;

    public ElementwiseBinaryOperator(ComputeNode left, ComputeNode right,
                                     FunctionRegistry.BinaryCalcFunction forwardFunction,
                                     FunctionRegistry.BinaryGradientFunction leftGradientFunction,
                                     FunctionRegistry.BinaryGradientFunction rightGradientFunction,
                                     String operatorName) {
        super(validateAndCreateResultStore(left, right, operatorName), left, right);
        setBroadcastFlags(left, right);
        this.left = left;
        this.right = right;
        this.forwardFunction = forwardFunction;
        this.leftGradientFunction = leftGradientFunction;
        this.rightGradientFunction = rightGradientFunction;
        this.operatorName = operatorName;
        computeResult();
    }

    private record Dimension(int numRows, int numCols){}

    private static MatrixStore validateAndCreateResultStore(ComputeNode left, ComputeNode right, String operatorName) {
        Dimension leftDims = new Dimension(left.numRows(), left.numCols());
        Dimension rightDims = new Dimension(right.numRows(), right.numCols());

        Dimension targetDims = new Dimension(Math.max(leftDims.numRows, rightDims.numRows), Math.max(leftDims.numCols, rightDims.numCols));

        if(targetDims.numRows != leftDims.numRows) {
            if (leftDims.numRows!=1) {
                throw new IllegalArgumentException("The "
                        + operatorName
                        + " operation is an elementwise binary operation and requires its inputs to be of the same dimensions or brodcastable."
                        + " Found mismatch ("+left.numRows()+", "+left.numCols()+") ("+right.numRows()+", "+right.numCols()+")");
            }
        }
        if(targetDims.numRows != rightDims.numRows) {
            if (rightDims.numRows!=1) {
                throw new IllegalArgumentException("The "
                        + operatorName
                        + " operation is an elementwise binary operation and requires its inputs to be of the same dimensions or brodcastable."
                        + " Found mismatch ("+left.numRows()+", "+left.numCols()+") ("+right.numRows()+", "+right.numCols()+")");
            }
        }
        if(targetDims.numCols != leftDims.numCols) {
            if (leftDims.numCols!=1) {
                throw new IllegalArgumentException("The "
                        + operatorName
                        + " operation is an elementwise binary operation and requires its inputs to be of the same dimensions or brodcastable."
                        + " Found mismatch ("+left.numRows()+", "+left.numCols()+") ("+right.numRows()+", "+right.numCols()+")");
            }
        }
        if(targetDims.numCols != rightDims.numCols) {
            if (rightDims.numCols!=1) {
                throw new IllegalArgumentException("The "
                        + operatorName
                        + " operation is an elementwise binary operation and requires its inputs to be of the same dimensions or brodcastable."
                        + " Found mismatch ("+left.numRows()+", "+left.numCols()+") ("+right.numRows()+", "+right.numCols()+")");
            }
        }

        return new MatrixStore(targetDims.numRows(), targetDims.numCols());
    }

    private void setBroadcastFlags(ComputeNode left, ComputeNode right) {
        Dimension leftDims = new Dimension(left.numRows(), left.numCols());
        Dimension rightDims = new Dimension(right.numRows(), right.numCols());

        Dimension targetDims = new Dimension(Math.max(leftDims.numRows, rightDims.numRows), Math.max(leftDims.numCols, rightDims.numCols));

        broadcastLeftRows = targetDims.numRows != leftDims.numRows && leftDims.numRows==1;
        broadcastRightRows = targetDims.numRows != rightDims.numRows && rightDims.numRows==1;
        broadcastLeftColumns = targetDims.numCols != leftDims.numCols && leftDims.numCols==1;
        broadcastRightColumns = targetDims.numCols != rightDims.numCols && rightDims.numCols==1;
    }

    @Override
    public void computeResult() {
        MatrixStore leftValues = left.getValues();
        MatrixStore rightValues = right.getValues();
        MatrixStore outputValues = getValues();
        float[]leftData = leftValues.backingArray();
        float[]rightData = rightValues.backingArray();
        float[]outputData = outputValues.backingArray();
        final int numRows = outputValues.numRows();
        final int numCols = outputValues.numRows();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                int leftIndex = leftValues.computeIndex(broadcastLeftRows?0:row, broadcastLeftColumns?0:col);
                int rightIndex = rightValues.computeIndex(broadcastRightRows?0:row, broadcastRightColumns?0:col);
                int outputIndex = outputValues.computeIndex(row, col);

                outputData[outputIndex] = (float)forwardFunction.result(leftData[leftIndex], rightData[rightIndex]);
            }
        }
    }

    @Override
    public void backpropogateGradients() {
        final boolean requiresGradient = requiresGradient();
        final boolean leftRequiresGradient = left.requiresGradient();
        final boolean rightRequiresGradient = right.requiresGradient();
        if (!requiresGradient) {
            //If this node does not require a gradient,
            // it cannot propogate it backward
            return;
        }
        if (!leftRequiresGradient && !rightRequiresGradient) {
            //If no child requires a gradient we need not propogate a gradient backward
            return;
        }
        MatrixStore leftValues = left.getValues();
        MatrixStore leftGradients = null;
        if (leftRequiresGradient) {
            leftGradients = left.getGradients();
        }
        MatrixStore rightValues = right.getValues();
        MatrixStore rightGradients = right.getGradients();
        if (rightRequiresGradient) {
            rightGradients = right.getGradients();
        }
        MatrixStore outputValues = getValues();
        MatrixStore outputGradients = getGradients();
        float[]leftData = leftValues.backingArray();
        float[]leftGradientData = null;
        if (leftRequiresGradient) {
            leftGradientData = leftGradients.backingArray();
        }
        float[]rightData = rightValues.backingArray();
        float[]rightGradientData = null;
        if (rightRequiresGradient) {
            rightGradientData = rightGradients.backingArray();
        }
        float[]outputData = outputValues.backingArray();
        float[]outputGradientData = null;
        outputGradientData = outputGradients.backingArray();

        final int numRows = outputValues.numRows();
        final int numCols = outputValues.numRows();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                int leftIndex = leftValues.computeIndex(broadcastLeftRows?0:row, broadcastLeftColumns?0:col);
                int rightIndex = rightValues.computeIndex(broadcastRightRows?0:row, broadcastRightColumns?0:col);
                int outputIndex = outputValues.computeIndex(row, col);
                int outputGradientIndex = outputGradients.computeIndex(row, col);

                if (leftRequiresGradient) {
                    int leftGradientIndex = leftGradients.computeIndex(broadcastLeftRows?0:row, broadcastLeftColumns?0:col);
                    leftGradientData[leftGradientIndex] += (float)leftGradientFunction.result(leftData[leftIndex], rightData[rightIndex], outputData[outputIndex], outputGradientData[outputGradientIndex]);
                }
                if (rightRequiresGradient) {
                    int rightGradientIndex = rightGradients.computeIndex(broadcastRightRows?0:row, broadcastRightColumns?0:col);
                    rightGradientData[rightGradientIndex] += (float)rightGradientFunction.result(leftData[leftIndex], rightData[rightIndex], outputData[outputIndex], outputGradientData[outputGradientIndex]);
                }
            }
        }
    }
}
