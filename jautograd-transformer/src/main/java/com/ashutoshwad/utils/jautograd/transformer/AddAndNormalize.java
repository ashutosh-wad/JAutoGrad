package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class AddAndNormalize {
    private Matrix scale;
    private Matrix shift;
    public AddAndNormalize(int featureSize) {
        scale = Matrix.create(1, featureSize, ()->1.0, true);
        shift = Matrix.create(1, featureSize, ()->0.0, true);
    }

    public Matrix apply(Matrix residual, Matrix input) {
        return residual.add(input).layerNorm(1).mul(scale).add(shift);
    }
}
