package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The compute node class is used to perform backpropogatable actions on matrices.
 * The resultant compute graph is not paralelizable or thread safe.
 */
public abstract class ComputeNode {
    private static final AtomicLong idFactory = new AtomicLong(0);

    /**
     * The id is used to identify causal relationships.
     * The idea is that as id's are provided in the order of creation, older id's must exist before the current node.
     * Due to this the forward pass need only execute all nodes in the dependencies in the ascending order of id's and
     * backpropogation would execute in descending order of id's,
     */
    private final long id;
    private final ComputeNode[]parents;
    private final MatrixStore values;
    private final MatrixStore gradients;
    private final boolean requiresGradient;
    private List<ComputeNode>nodes;

    protected ComputeNode(MatrixStore values, boolean requiresGradient) {
        this.id = idFactory.getAndIncrement();
        this.parents = null;
        this.values = values;

        this.requiresGradient = requiresGradient;
        if (requiresGradient) {
            this.gradients = new MatrixStore(values.numRows(), values.numCols());
        } else {
            this.gradients = null;
        }
    }

    protected ComputeNode(MatrixStore values, ComputeNode...parents) {
        this.id = idFactory.getAndIncrement();
        this.parents = parents;
        this.values = values;

        //If any of the parents need gradient computation, this node must need it as well
        boolean temp = false;
        for (ComputeNode parent : parents) {
            temp = temp || parent.requiresGradient;
        }
        this.requiresGradient = temp;
        if (requiresGradient) {
            this.gradients = new MatrixStore(values.numRows(), values.numCols());
        } else {
            this.gradients = null;
        }
    }

    protected ComputeNode[] getParents() {
        return parents;
    }

    public long getId() {
        return id;
    }

    public MatrixStore getValues() {
        return values;
    }

    public MatrixStore getGradients() {
        return gradients;
    }

    public boolean requiresGradient() {
        return requiresGradient;
    }

    public int numCols() {
        return getValues().numCols();
    }

    public int numRows() {
        return getValues().numRows();
    }

    public float getValue(int row, int column) {
        return getValues().get(row, column);
    }

    public void setValue(int row, int column, float value) {
        getValues().set(row, column, value);
    }

    public void addValue(int row, int column, float value) {
        getValues().add(row, column, value);
    }

    public float getGradient(int row, int column) {
        return getGradients().get(row, column);
    }

    public void setGradient(int row, int column, float value) {
        getGradients().set(row, column, value);
    }

    public void addGradient(int row, int column, float value) {
        getGradients().add(row, column, value);
    }

    public void fillValues(float value) {
        getValues().fill(value);
    }

    public void fillGradients(float value) {
        if(!requiresGradient) {
            return;
        }
        getGradients().fill(value);
    }

    private void fetchNodes(Map<Long, ComputeNode> nodeMap) {
        //This method works as there is
        // no possibility of cyclic graphs
        // as per design
        nodeMap.put(id, this);
        if (null == parents) {
            return;
        }
        for (ComputeNode parent : parents) {
            parent.fetchNodes(nodeMap);
        }
    }

    private void computeNodes() {
        if (nodes != null) {
            //No need to re-compute
            return;
        }
        TreeMap<Long, ComputeNode>nodeMap = new TreeMap<>();
        fetchNodes(nodeMap);
        nodes = new ArrayList<>(nodeMap.size());
        nodeMap.forEach((k, v) -> nodes.add(v));
    }

    public final void forward() {
        computeNodes();
        int length = nodes.size();
        for (int i = 0; i < length; i++) {
            nodes.get(i).computeResult();
        }
    }

    public final void zeroGrad() {
        computeNodes();
        int length = nodes.size();
        for (int i = 0; i < length; i++) {
            nodes.get(i).fillGradients(0);
        }
    }

    public final void zeroGradAndForward() {
        computeNodes();
        int length = nodes.size();
        for (int i = 0; i < length; i++) {
            nodes.get(i).fillGradients(0);
            nodes.get(i).computeResult();
        }
    }

    public final void backward() {
        backward(true);
    }
    public final void backward(boolean setGradToOne) {
        computeNodes();
        if (requiresGradient) {
            if(setGradToOne) {
                gradients.fill(1.0f);
            }
        }
        int length = nodes.size();
        for (int i = length-1; i >=0; i--) {
            ComputeNode temp = nodes.get(i);
            if (temp.requiresGradient) {
                temp.backpropogateGradients();
            }
        }
    }

    abstract public void computeResult();
    abstract public void backpropogateGradients();
}
