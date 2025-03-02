package com.ashutoshwad.utils.jautograd.compute;

interface GradientFunction {
    public double apply(ComputeNode output, ComputeNode left, ComputeNode right, boolean isInvokerLeft);
}
