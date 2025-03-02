package com.ashutoshwad.utils.jautograd.compute;

interface ComputeFunction {
    public void apply(ComputeNode left, ComputeNode right, ComputeNode target);
}
