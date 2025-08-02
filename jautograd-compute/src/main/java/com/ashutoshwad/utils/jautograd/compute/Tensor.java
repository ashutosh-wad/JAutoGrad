package com.ashutoshwad.utils.jautograd.compute;

import java.util.Arrays;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;

public class Tensor {
    private ComputeNode[]nodes;
    private final int[] shape;
    private final int[] stride;
    private final int size;
    private final int rank;

    private Tensor(int[]shape, Supplier<Double>initialValues) {
        //Validate the shape
        if(null == shape || shape.length == 0) {
            throw new IllegalArgumentException("Number of shape cannot be empty");
        }
        for (int i = shape.length - 1; i >= 0; i--) {
            if(shape[i] == 0) {
                throw new IllegalArgumentException("No dimension can be 0");
            }
        }
        //Set the shape
        this.shape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            this.shape[i] = shape[i];
        }
        //Validate supplier
        if(null == initialValues) {
            initialValues = () -> 0.0;
        }
        //Set the stride
        stride = computeStrides(shape);
        size = computeSize(shape);
        rank = shape.length;
        //Set the nodes
        nodes = new ComputeNode[size];
        for (int i = 0; i < size; i++) {
            nodes[i] = new ComputeNode(initialValues.get());
        }
    }

    private int computeSize(int[] shape) {
        int size = 1;
        for (int i = 0; i < shape.length; i++) {
            size = size * shape[i];
        }
        return size;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private boolean shapeMatches(int[] shape1, int[] shape2) {
        if(shape1.length != shape2.length) {
            return false;
        }
        for (int i = shape1.length - 1; i >= 0; i--) {
            if(shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    }

    private int[] getBroadcastSize(int[] shape1, int[] shape2) {
        //Ensure lengths are same
        if(shape1.length != shape2.length) {
            if(shape1.length < shape2.length) {
                int[]temp = shape1;
                shape1 = shape2;
                shape2 = temp;
            }
            int[]temp = new int[shape1.length];
            for (int i = temp.length - 1; i >= 0; i--) {
                int j = shape2.length - (temp.length - i - 1);
                if(j>=0) {
                    temp[i] = shape2[j];
                } else {
                    temp[i] = 1;
                }
            }
            shape2 = temp;
        }
        //Now either dimension has to be 1 or equal
        for (int i = shape1.length - 1; i >= 0; i--) {
            if(shape1[i] == shape2[i]) {
                continue;
            }
            if(shape1[i] == 1 || shape2[i] == 1) {
                continue;
            }
            throw new IllegalArgumentException("Provided dimensions are not broadcastable");
        }
        //Now compute the broadcast array shape
        int[]bShape = new int[shape1.length];
        for (int i = 0; i < bShape.length; i++) {
            bShape[i] = Math.max(shape1[i], shape2[i]);
        }
        return bShape;
    }

    private Tensor broadcast(Tensor t, int[] bShape) {
        Tensor tRes = new Tensor(bShape, ()->null);
        Optional<int[]>index = Optional.of(new int[bShape.length]);
        do {
            tRes.set(index.get(), t.get(maskedBroadcastIndex(index.get(), t.shape)));
            index = incrementIndex(index.get(), bShape);
        } while (index.isPresent());
        return tRes;
    }

    private int[] maskedBroadcastIndex(int[]index, int[]shape) {
        int[]bIndex = new int[shape.length];
        for (int i = bIndex.length - 1; i >= 0 ; i--) {
            int j = index.length - (bIndex.length - i);
            if(shape[i]==1) {
                bIndex[i] = 0;
            } else {
                bIndex[i] = index[j];
            }
        }
        return bIndex;
    }

    private Optional<int[]> incrementIndex(int[]index, int[]length) {
        int zeroCount = 0;
        for (int i = index.length - 1; i >= 0 ; i--) {
            index[i]++;
            if(index[i] == length[i]) {
                index[i] = 0;
                zeroCount++;
            } else {
                break;
            }
        }
        if(zeroCount == length.length) {
            return Optional.empty();
        } else {
            return Optional.of(index);
        }
    }

    private int calculateNodeIndex(int[]loc) {
        if(loc.length != shape.length) {
            throw new ArrayIndexOutOfBoundsException("Provided index does not match tensor shape.");
        }
        int index = 0;
        for (int i = 0; i < loc.length; i++) {
            index += loc[i] * stride[i];
        }
        return index;
    }

    private Tensor op(Tensor t1, Tensor t2, BiFunction<ComputeNode, ComputeNode, ComputeNode> function) {
        if(!shapeMatches(t1.shape, t2.shape)) {
            int[]bShape = null;
            try {
                bShape = getBroadcastSize(t1.shape, t2.shape);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("The tensors are incompatible for mathematical operations. Broadcast has failed" );
            }
            if(!shapeMatches(bShape, t1.shape)) {
                t1 = broadcast(t1, bShape);
            }
            if(!shapeMatches(bShape, t2.shape)) {
                t2 = broadcast(t2, bShape);
            }
        }
        Tensor result = Tensor.createTensor(t1.shape, ()->null);
        for (int i = 0; i < t1.nodes.length; i++) {
            result.nodes[i] = function.apply(t1.nodes[i], t2.nodes[i]);
        }
        return result;
    }

    public static final Tensor createTensor(int[]shape, Supplier<Double>initialValues) {
        return new Tensor(shape, initialValues);
    }

    public static final Tensor zeroes(int[]shape) {
        return new Tensor(shape, ()->0.0);
    }

    //LLM generated methods
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape))
                .append(", size=").append(size)
                .append(", rank=").append(rank)
                .append(")\n");

        if (rank == 1) {
            sb.append("[");
            for (int i = 0; i < Math.min(size, 10); i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.3f", nodes[i].getValue()));
            }
            if (size > 10) sb.append(", ...");
            sb.append("]");
        } else if (rank == 2) {
            sb.append("[");
            for (int i = 0; i < Math.min(shape[0], 5); i++) {
                if (i > 0) sb.append(",\n ");
                sb.append("[");
                for (int j = 0; j < Math.min(shape[1], 5); j++) {
                    if (j > 0) sb.append(", ");
                    sb.append(String.format("%.3f", nodes[i * stride[0] + j * stride[1]].getValue()));
                }
                if (shape[1] > 5) sb.append(", ...");
                sb.append("]");
            }
            if (shape[0] > 5) sb.append(",\n ...");
            sb.append("]");
        } else {
            sb.append("Data preview: [");
            for (int i = 0; i < Math.min(size, 10); i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.3f", nodes[i].getValue()));
            }
            if (size > 10) sb.append(", ...");
            sb.append("]");
        }

        return sb.toString();
    }

    //API methods start here
    public ComputeNode get(int[]loc) {
        return nodes[calculateNodeIndex(loc)];
    }
    public ComputeNode set(int[]loc, ComputeNode node) {
        int index = calculateNodeIndex(loc);
        ComputeNode old = nodes[index];
        nodes[index] = node;
        return old;
    }
}
