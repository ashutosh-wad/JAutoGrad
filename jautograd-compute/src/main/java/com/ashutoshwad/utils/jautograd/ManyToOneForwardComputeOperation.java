package com.ashutoshwad.utils.jautograd;

class ManyToOneForwardComputeOperation extends ForwardComputeOperation {
    ManyToOneForwardComputeOperation(ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
    }

    @Override
    protected void perform() {
        //No need to do anything
    }
}
