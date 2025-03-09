package com.ashutoshwad.utils.jautograd.compute;

import static com.ashutoshwad.utils.jautograd.compute.FunctionRegistry.*;
import java.util.concurrent.atomic.AtomicLong;

public class ComputeNode {
    private static final AtomicLong generator = new AtomicLong(0);
    private final long id;
    private final ComputeNode left;
    private final ComputeNode right;
    private final ComputeFunction calcFunction;
    private final GradientFunction gradientFunction;
    private double value;
    private double gradient;
    private ComputeNode[] dependencies;
    private boolean[] dependencyDirections;

    public ComputeNode(double value) {
        this(null, null, NOOP_COMPUTE, NOOP_GRADIENT);
        this.value = value;
        calc();
    }

    protected ComputeNode(ComputeNode left, ComputeNode right, ComputeFunction calcFunction, GradientFunction gradientFunction) {
        this.id = generator.getAndIncrement();
        this.left = left;
        this.right = right;
        this.calcFunction = calcFunction;
        this.gradientFunction = gradientFunction;
        if (null != left) {
            left.addDependency(this, true);
        }
        if (null != right) {
            right.addDependency(this, false);
        }
        calc();
    }

    public long getId() {
        return id;
    }

    public double getValue() {
        return this.value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getGradient() {
        return gradient;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    /*
    protected boolean isUnaryOperator() {
        return left != null && right == null;
    }

    protected boolean isBinaryOperator() {
        return left != null && right != null;
    }
    */

    protected void addDependency(ComputeNode dependency, boolean isLeft) {
        if (null == dependencies) {
            dependencies = new ComputeNode[1];
            dependencyDirections = new boolean[1];
            dependencies[0] = dependency;
            dependencyDirections[0] = isLeft;
        } else {
            ComputeNode[] newDependencies = new ComputeNode[dependencies.length + 1];
            boolean[] newDependencyDirections = new boolean[dependencies.length + 1];

            System.arraycopy(dependencies, 0, newDependencies, 0, dependencies.length);
            System.arraycopy(dependencyDirections, 0, newDependencyDirections, 0, dependencies.length);

            newDependencies[newDependencies.length - 1] = dependency;
            newDependencyDirections[newDependencies.length - 1] = isLeft;

            dependencies = newDependencies;
            dependencyDirections = newDependencyDirections;
        }
    }

    public void calc() {
        calcFunction.apply(left, right, this);
    }

    private double computePartialGradient(boolean isInvokerLeft) {
        return gradientFunction.apply(this, left, right, isInvokerLeft);
    }

    public void computeGradient() {
        if (null == dependencies || dependencies.length == 0) {
            //This indicates that this is an output node, so gradient will be 1
            gradient = 1;
        } else {
            gradient = 0;
            for (int i = 0; i < dependencies.length; i++) {
                gradient += dependencies[i].computePartialGradient(dependencyDirections[i]);
            }
        }
    }

    protected void visit(ComputeNodeVisitor visitor) {
        visitor.visit(this, left, right, dependencies);
    }

    public JAutogradExecutor createExecutor() {
        ComputeNodeVisitor visitor = new ComputeNodeVisitor();
        visitor.visit(this, left, right, dependencies);
        JAutogradExecutor executor = new JAutogradExecutor(visitor.prepareBatches());
        return executor;
    }

    public JAutogradExecutor createExecutor(int numThreads) {
        ComputeNodeVisitor visitor = new ComputeNodeVisitor();
        visitor.visit(this, left, right, dependencies);
        JAutogradExecutor executor = new JAutogradExecutor(numThreads, visitor.prepareBatches());
        return executor;
    }

    public ComputeNode add(ComputeNode o) {
        return new ComputeNode(this, o, ADD, ADD_GRAD);
    }

    public ComputeNode sub(ComputeNode o) {
        return new ComputeNode(this, o, SUB, SUB_GRAD);
    }

    public ComputeNode mul(ComputeNode o) {
        return new ComputeNode(this, o, MUL, MUL_GRAD);
    }

    public ComputeNode div(ComputeNode o) {
        return new ComputeNode(this, o, DIV, DIV_GRAD);
    }

    public ComputeNode pow(ComputeNode o) {
        return new ComputeNode(this, o, POW, POW_GRAD);
    }

    public ComputeNode max(ComputeNode o) {
        return new ComputeNode(this, o, MAX, NOOP_GRADIENT);
    }

    public ComputeNode min(ComputeNode o) {
        return new ComputeNode(this, o, MIN, NOOP_GRADIENT);
    }

    public ComputeNode sin() {
        return new ComputeNode(this, null, SIN, SIN_GRAD);
    }

    public ComputeNode cos() {
        return new ComputeNode(this, null, COS, COS_GRAD);
    }

    public ComputeNode tan() {
        return new ComputeNode(this, null, TAN, TAN_GRAD);
    }

    public ComputeNode sinh() {
        return new ComputeNode(this, null, SINH, SINH_GRAD);
    }

    public ComputeNode cosh() {
        return new ComputeNode(this, null, COSH, COSH_GRAD);
    }

    public ComputeNode tanh() {
        return new ComputeNode(this, null, TANH, TANH_GRAD);
    }

    public ComputeNode relu() {
        return new ComputeNode(this, null, RELU, RELU_GRAD);
    }

    public ComputeNode leakyRelu(double negSlope) {
        return new ComputeNode(this, null, LEAKY_RELU, LEAKY_RELU_GRAD);
    }

    public ComputeNode exp() {
        return new ComputeNode(this, null, EXP, EXP_GRAD);
    }

    public ComputeNode ln() {
        return new ComputeNode(this, null, LN, LN_GRAD);
    }

    public ComputeNode log() {
        return new ComputeNode(this, null, LOG, LOG_GRAD);
    }
}
