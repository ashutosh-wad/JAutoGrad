package com.ashutoshwad.utils.jautograd.compute.model;

import com.ashutoshwad.utils.jautograd.compute.ComputeNode;
import com.ashutoshwad.utils.jautograd.compute.Matrix;

public class RotaryPositionalEncoding implements TransformerStage {
    private final int dk;

    public RotaryPositionalEncoding(int dk) {
        this.dk = dk;
    }

    @Override
    public Matrix apply(Matrix input) {
        if(input.columnLength()%2!=0) {
            throw new IllegalArgumentException("RoPE expects feature dimensions to be divisible by 2");
        }
        //Create a copy of input so that setter operations do not affect the original matrix
        Matrix output = new Matrix(input);
        for (int m = 0; m < output.rowLength(); m++) {
            for (int i = 0; i < output.columnLength()/2; i++) {
                Matrix featurePair = new Matrix(new ComputeNode[][]{
                        {output.get(m, i*2), output.get(m, i*2 + 1)}
                });
                Matrix rotationMatrix = createRotationMatrix(m, i);
                featurePair = featurePair.matmul(rotationMatrix);
                output.set(m, i*2, featurePair.get(0, 0));
                output.set(m, i*2 + 1, featurePair.get(0, 1));
            }
        }
        return output;
    }

    private Matrix createRotationMatrix(int m, int i) {
        double angleOfRotation = calculateAngleOfRotation(m, i);
        double cos = Math.cos(angleOfRotation);
        double sin = Math.sin(angleOfRotation);
        ComputeNode[][]rot = new ComputeNode[2][2];
        rot[0][0] = new ComputeNode(cos);
        rot[0][1] = new ComputeNode(-1 * sin);
        rot[1][0] = new ComputeNode(sin);
        rot[1][1] = rot[0][0];
        return new Matrix(rot);
    }

    private double calculateAngleOfRotation(int m, int i) {
        double base = 0.0001;
        double exp = 2.0 * i / dk;
        return m * Math.pow(base, exp);
    }
}
