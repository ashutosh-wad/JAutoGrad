package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class AddAndNormalize {
    private Matrix scale;
    private Matrix shift;
    public AddAndNormalize(int featureSize) {
        scale = Matrix.create(1, featureSize, ()->1.0f, true);
        shift = Matrix.create(1, featureSize, ()->0.0f, true);
    }

    public Matrix apply(Matrix residual, Matrix input) {
        return residual.add(input).layerNorm(1).mul(scale).add(shift);
    }
}
