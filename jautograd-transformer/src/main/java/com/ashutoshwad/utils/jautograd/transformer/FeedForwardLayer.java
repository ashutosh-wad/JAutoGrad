package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class FeedForwardLayer {
    private final Matrix layer1;
    private final Matrix bias1;
    private final Matrix layer2;
    private final Matrix bias2;

    public FeedForwardLayer(int featureSize) {
        layer1 = Matrix.createXavierGlorotInitializedMatrix(featureSize, 4 * featureSize, true);
        bias1 = Matrix.create(1, 4 * featureSize, true);

        layer2 = Matrix.createXavierGlorotInitializedMatrix(4 * featureSize, featureSize, true);
        bias2 = Matrix.create(1, featureSize, true);
    }

    public Matrix apply(Matrix input, boolean dropout, double dropoutRate) {
        if (dropout) {
            return input.matmul(layer1).add(bias1).swish().dropout(dropoutRate).matmul(layer2).add(bias2);
        } else {
            return input.matmul(layer1).add(bias1).swish().matmul(layer2).add(bias2);
        }
    }
}
