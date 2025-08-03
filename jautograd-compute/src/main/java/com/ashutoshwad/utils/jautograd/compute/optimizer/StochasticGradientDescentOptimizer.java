package com.ashutoshwad.utils.jautograd.compute.optimizer;

import com.ashutoshwad.utils.jautograd.compute.ComputeNode;

public class StochasticGradientDescentOptimizer implements LearningOptimizer {
    private final ComputeNode computeNode;

    public StochasticGradientDescentOptimizer(ComputeNode computeNode) {
        this.computeNode = computeNode;
    }

    public void learn(double learningRate) {
        if(computeNode.isTrainable()) {
            computeNode.setValue(computeNode.getValue() + learningRate * (computeNode.getGradient() * -1));
        }
    }
}
