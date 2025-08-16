package com.ashutoshwad.utils.jautograd;

class ManyToOneBackwardComputeOperation extends BackwardComputeOperation {
    ManyToOneBackwardComputeOperation(BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
    }

    @Override
    protected void perform() {
        //No need to do anything
    }
}
