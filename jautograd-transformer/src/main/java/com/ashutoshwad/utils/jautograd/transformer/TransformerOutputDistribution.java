package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class TransformerOutputDistribution {
    private LayerNormalizer layerNorm;
    private Matrix sampler;

    public TransformerOutputDistribution(int featureSize, int tokenSize) {
        layerNorm = new LayerNormalizer(featureSize);
        sampler = Matrix.createXavierGlorotInitializedMatrix(featureSize, tokenSize, true);
    }

    public Matrix apply(Matrix input) {
        return layerNorm.apply(input).matmul(sampler).softmax(1);
    }
}
