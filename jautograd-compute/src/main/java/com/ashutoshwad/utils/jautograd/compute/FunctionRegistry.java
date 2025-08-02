package com.ashutoshwad.utils.jautograd.compute;

class FunctionRegistry {
    public static final ComputeFunction NOOP_COMPUTE = (l, r, t) -> {
    };
    public static final ComputeFunction ADD = (l, r, t) -> t.setValue(l.getValue() + r.getValue());
    public static final ComputeFunction SUB = (l, r, t) -> t.setValue(l.getValue() - r.getValue());
    public static final ComputeFunction DIV = (l, r, t) -> t.setValue(l.getValue() / r.getValue());
    public static final ComputeFunction MUL = (l, r, t) -> t.setValue(l.getValue() * r.getValue());
    public static final ComputeFunction POW = (l, r, t) -> t.setValue(Math.pow(l.getValue(), r.getValue()));
    public static final ComputeFunction SQRT = (l, r, t) -> t.setValue(Math.sqrt(l.getValue()));
    public static final ComputeFunction MAX = (l, r, t) -> t.setValue((Math.max(l.getValue(), r.getValue())));
    public static final ComputeFunction MIN = (l, r, t) -> t.setValue((Math.min(l.getValue(), r.getValue())));
    public static final ComputeFunction SIN = (l, r, t) -> t.setValue(Math.sin(l.getValue()));
    public static final ComputeFunction COS = (l, r, t) -> t.setValue(Math.cos(l.getValue()));
    public static final ComputeFunction TAN = (l, r, t) -> t.setValue(Math.tan(l.getValue()));
    public static final ComputeFunction SINH = (l, r, t) -> t.setValue(Math.sinh(l.getValue()));
    public static final ComputeFunction COSH = (l, r, t) -> t.setValue(Math.cosh(l.getValue()));
    public static final ComputeFunction TANH = (l, r, t) -> t.setValue(Math.tanh(l.getValue()));
    public static final ComputeFunction RELU = (l, r, t) -> t.setValue((l.getValue() >= 0) ? l.getValue() : 0);
    public static final ComputeFunction LEAKY_RELU = (l, r, t) -> t.setValue((l.getValue() >= 0) ? l.getValue() : 0.1 * l.getValue());
    public static final ComputeFunction EXP = (l, r, t) -> t.setValue(Math.exp(l.getValue()));
    public static final ComputeFunction LN = (l, r, t) -> t.setValue(Math.log(l.getValue()));
    public static final ComputeFunction LOG = (l, r, t) -> t.setValue(Math.log10(l.getValue()));
    public static final ComputeFunction SIGMOID = (l, r, t) -> {
        double x = l.getValue();
        double sigmoid = 1 / ( 1 + Math.exp(-1 * x) );
        t.setValue(sigmoid);
    };
    public static final ComputeFunction SIMPLE_SWISH = (l, r, t) -> {
        double x = l.getValue();
        double result = x / (1 + Math.exp(-1 * x));
        t.setValue(result);
    };

    public static final GradientFunction NOOP_GRADIENT = (o, l, r, isInvokerLeft) -> 0;
    public static final GradientFunction ADD_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient();
    };
    public static final GradientFunction SUB_GRAD = (o, l, r, isInvokerLeft) -> {
        return isInvokerLeft ? o.getGradient() : -1 * o.getGradient();
    };
    public static final GradientFunction DIV_GRAD = (o, l, r, isInvokerLeft) -> {
        return isInvokerLeft ? o.getGradient() * (1 / r.getValue()) : o.getGradient() * (l.getValue() * (-1 / (r.getValue() * r.getValue())));
    };
    public static final GradientFunction MUL_GRAD = (o, l, r, isInvokerLeft) -> {
        return isInvokerLeft ? o.getGradient() * r.getValue() : o.getGradient() * l.getValue();
    };
    public static final GradientFunction POW_GRAD = (o, l, r, isInvokerLeft) -> {
        return isInvokerLeft ? o.getGradient() * r.getValue() * Math.pow(l.getValue(), (r.getValue() - 1)) : o.getGradient() * o.getValue() * Math.log(l.getValue());
    };
    public static final GradientFunction SQRT_GRAD = (o, l, r, isInvokerLeft) -> {
        double sqrt_x = o.getValue();
        double derivative = 1.0 / (2.0 * sqrt_x);
        return o.getGradient() * derivative;
    };
    public static final GradientFunction MAX_GRAD = (o, l, r, isInvokerLeft) -> {
        double diff = Math.abs(l.getValue() - r.getValue());
        if (diff < 1e-8) {
            return 0.5 * o.getGradient();
        }
        if (l.getValue() > r.getValue()) {
            return isInvokerLeft ? o.getGradient() : 0;
        } else {
            return isInvokerLeft ? 0 : o.getGradient();
        }
    };
    public static final GradientFunction MIN_GRAD = (o, l, r, isInvokerLeft) -> {
        double diff = Math.abs(l.getValue() - r.getValue());
        if (diff < 1e-8) {
            return 0.5 * o.getGradient();
        }
        if (l.getValue() > r.getValue()) {
            return isInvokerLeft ? 0 : o.getGradient();
        } else {
            return isInvokerLeft ? o.getGradient() : 0;
        }
    };
    public static final GradientFunction SIN_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * Math.cos(l.getValue());
    };
    public static final GradientFunction COS_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * -1 * Math.sin(l.getValue());
    };
    public static final GradientFunction TAN_GRAD = (o, l, r, isInvokerLeft) -> {
        double temp = Math.cos(l.getValue());
        temp = temp * temp;
        return o.getGradient() / temp;
    };
    public static final GradientFunction SINH_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * Math.cosh(l.getValue());
    };
    public static final GradientFunction COSH_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * Math.sinh(l.getValue());
    };
    public static final GradientFunction TANH_GRAD = (o, l, r, isInvokerLeft) -> {
        double temp = Math.cosh(l.getValue());
        temp = temp * temp;
        return o.getGradient() / temp;
    };
    public static final GradientFunction RELU_GRAD = (o, l, r, isInvokerLeft) -> {
        double temp = l.getValue();
        temp = (temp == -0.0) ? 0.0 : temp;
        return o.getGradient() * ((temp >= 0) ? 1 : 0);
    };
    public static final GradientFunction LEAKY_RELU_GRAD = (o, l, r, isInvokerLeft) -> {
        double temp = l.getValue();
        temp = (temp == -0.0) ? 0.0 : temp;
        return o.getGradient() * ((temp >= 0) ? 1 : 0.1);
    };
    public static final GradientFunction EXP_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * o.getValue();
    };
    public static final GradientFunction LN_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * (1 / l.getValue());
    };
    private static final double LN10 = Math.log(10);
    public static final GradientFunction LOG_GRAD = (o, l, r, isInvokerLeft) -> {
        return o.getGradient() * (1 / (LN10 * l.getValue()));
    };
    public static final GradientFunction SIGMOID_GRAD = (o, l, r, isInvokerLeft) -> {
        double sigmoid = o.getValue();
        double dSigmoid = sigmoid * (1 - sigmoid);
        return o.getGradient() * dSigmoid;
    };
    public static final GradientFunction SIMPLE_SWISH_GRAD = (o, l, r, isInvokerLeft) -> {
        double x = l.getValue();
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));
        double swishGrad = sigmoid + x * sigmoid * (1.0 - sigmoid);
        return o.getGradient() * swishGrad;
    };
}
