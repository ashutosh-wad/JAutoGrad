package com.ashutoshwad.utils.jautograd.compute.model;

import com.ashutoshwad.utils.jautograd.compute.Matrix;

public interface TransformerStage {
    public Matrix apply(Matrix input);
}
