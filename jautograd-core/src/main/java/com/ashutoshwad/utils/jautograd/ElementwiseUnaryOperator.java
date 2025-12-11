package com.ashutoshwad.utils.jautograd;

public class ElementwiseUnaryOperator extends Matrix {
    private final FunctionRegistry.UnaryCalcFunction forwardFunction;
    private final FunctionRegistry.UnaryGradientFunction gradientFunction;
    private final ComputeNode input;
    public final String operatorName;

    public ElementwiseUnaryOperator(ComputeNode input, FunctionRegistry.UnaryCalcFunction forwardFunction, FunctionRegistry.UnaryGradientFunction gradientFunction, String operatorName) {
        super(new MatrixStore(input.getValues().numRows(), input.getValues().numCols()), input);
        this.input = input;
        this.forwardFunction = forwardFunction;
        this.gradientFunction = gradientFunction;
        this.operatorName = operatorName;
        computeResult();
    }

    @Override
    public void computeResult() {
        MatrixStore inputValues = input.getValues();
        MatrixStore outputValues = getValues();
        float[]inputData = inputValues.backingArray();
        float[]outputData = outputValues.backingArray();
        final int numRows = inputValues.numRows();
        final int numCols = inputValues.numRows();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                outputData[outputValues.computeIndex(row, col)] = (float)forwardFunction.result(inputData[inputValues.computeIndex(row, col)]);
            }
        }
    }

    @Override
    public void backpropogateGradients() {
        final boolean requiresGradient = requiresGradient();
        final boolean inputRequiresGradient = input.requiresGradient();
        if (!requiresGradient) {
            //If this node does not require a gradient,
            // it cannot propogate it backward
            return;
        }
        if (!inputRequiresGradient) {
            //If no child requires a gradient we need not propogate a gradient backward
            return;
        }

        MatrixStore inputValues = input.getValues();
        MatrixStore inputGradients = input.getGradients();
        MatrixStore outputValues = getValues();
        MatrixStore outputGradients = getGradients();
        float[]inputData = inputValues.backingArray();
        float[]inputGradientData = inputGradients.backingArray();
        float[]outputData = outputValues.backingArray();
        float[]outputGradientData = outputGradients.backingArray();
        final int numRows = inputValues.numRows();
        final int numCols = inputValues.numRows();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                int inputIndex = inputValues.computeIndex(row, col);
                int inputGradientIndex = inputGradients.computeIndex(row, col);
                int outputIndex = outputValues.computeIndex(row, col);
                int outputGradientIndex = outputGradients.computeIndex(row, col);

                inputGradientData[inputGradientIndex] += (float) gradientFunction.result(inputData[inputIndex],
                                                                                outputData[outputIndex],
                                                                                outputGradientData[outputGradientIndex]);
            }
        }
    }
}
