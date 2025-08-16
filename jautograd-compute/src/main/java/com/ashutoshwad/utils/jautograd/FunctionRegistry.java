package com.ashutoshwad.utils.jautograd;

class FunctionRegistry {
    private static final double EPSILON = 1e-9;

    private static double nanGuard(double x) {
        if (Math.abs(x) < 1e-12) {
            return x >= 0 ? 1e-12 : -1e-12;
        } else {
            return x;
        }
    }

    public static final BinaryCalcFunction ADD = (l, r) -> l + r;
    public static final BinaryCalcFunction SUB = (l, r) -> l - r;
    public static final BinaryCalcFunction DIV = (l, r) -> l / r;
    public static final BinaryCalcFunction MUL = (l, r) -> l * r;
    public static final BinaryCalcFunction POW = Math::pow;
    public static final UnaryCalcFunction SQRT = Math::sqrt;
    public static final BinaryCalcFunction MAX = Math::max;
    public static final BinaryCalcFunction MIN = Math::min;
    public static final UnaryCalcFunction SIN = Math::sin;
    public static final UnaryCalcFunction COS = Math::cos;
    public static final UnaryCalcFunction TAN = Math::tan;
    public static final UnaryCalcFunction SINH = Math::sinh;
    public static final UnaryCalcFunction COSH = Math::cosh;
    public static final UnaryCalcFunction TANH = Math::tanh;
    public static final UnaryCalcFunction RELU = l -> (l >= 0) ? l : 0;
    public static final UnaryCalcFunction LEAKY_RELU = l -> (l >= 0) ? l : 0.1 * l;
    public static final UnaryCalcFunction EXP = Math::exp;
    public static final UnaryCalcFunction LN = Math::log;
    public static final UnaryCalcFunction LOG = Math::log10;
    public static final UnaryCalcFunction SIGMOID = l -> 1 / ( 1 + Math.exp(-1 * l));
    public static final UnaryCalcFunction SIMPLE_SWISH = l -> l / (1 + Math.exp(-1 * l));

    public static final BinaryGradientFunction ADD_GRAD_LEFT = (l, r, o, oGrad) -> oGrad;
    public static final BinaryGradientFunction ADD_GRAD_RIGHT = (l, r, o, oGrad) -> oGrad;

    public static final BinaryGradientFunction SUB_GRAD_LEFT = (l, r, o, oGrad) -> oGrad;
    public static final BinaryGradientFunction SUB_GRAD_RIGHT = (l, r, o, oGrad) -> -1 * oGrad;

    public static final BinaryGradientFunction DIV_GRAD_LEFT = (l, r, o, oGrad) -> oGrad / nanGuard(r);
    public static final BinaryGradientFunction DIV_GRAD_RIGHT = (l, r, o, oGrad) -> {
        double temp = nanGuard(r);
        return oGrad * (l * (-1 / (temp * temp)));
    };

    public static final BinaryGradientFunction MUL_GRAD_LEFT = (l, r, o, oGrad) -> oGrad * r;
    public static final BinaryGradientFunction MUL_GRAD_RIGHT = (l, r, o, oGrad) -> oGrad * l;

    public static final BinaryGradientFunction POW_GRAD_LEFT = (l, r, o, oGrad) -> oGrad * r * Math.pow(l, (r - 1));
    public static final BinaryGradientFunction POW_GRAD_RIGHT = (l, r, o, oGrad) -> oGrad * o * Math.log(l);

    public static final UnaryGradientFunction SQRT_GRAD = (l, o, oGrad) -> {
        double sqrt_x = Math.max(o, 1e-12);
        double derivative = 1.0 / (2.0 * sqrt_x);
        return oGrad * derivative;
    };

    public static final BinaryGradientFunction MAX_GRAD_LEFT = (l, r, o, oGrad) -> {
        if (l>r+EPSILON) {
            return oGrad;
        } else if(Math.abs(l-r)<=EPSILON) {
            return oGrad * 0.5;
        } else {
            return 0;
        }
    };
    public static final BinaryGradientFunction MAX_GRAD_RIGHT = (l, r, o, oGrad) -> {
        if(r>l+EPSILON) {
            return oGrad;
        } else if (Math.abs(l-r)<=EPSILON) {
            return 0.5 * oGrad;
        } else {
            return 0;
        }
    };

    public static final BinaryGradientFunction MIN_GRAD_LEFT = (l, r, o, oGrad) -> {
        if(l<r+EPSILON) {
            return oGrad;
        } else if (Math.abs(l-r)<=EPSILON) {
            return 0.5 * oGrad;
        } else {
            return 0;
        }
    };
    public static final BinaryGradientFunction MIN_GRAD_RIGHT = (l, r, o, oGrad) -> {
        if(r<l+EPSILON) {
            return oGrad;
        } else if (Math.abs(l-r)<=EPSILON) {
            return 0.5 * oGrad;
        } else {
            return 0;
        }
    };

    public static final UnaryGradientFunction SIN_GRAD = (l, o, oGrad) -> oGrad * Math.cos(l);
    public static final UnaryGradientFunction COS_GRAD = (l, o, oGrad) -> oGrad * -1 * Math.sin(l);
    public static final UnaryGradientFunction TAN_GRAD = (l, o, oGrad) -> oGrad * (1 + o * o);

    public static final UnaryGradientFunction SINH_GRAD = (l, o, oGrad) -> oGrad * Math.cosh(l);
    public static final UnaryGradientFunction COSH_GRAD = (l, o, oGrad) -> oGrad * Math.sinh(l);
    public static final UnaryGradientFunction TANH_GRAD = (l, o, oGrad) -> oGrad * (1 - o * o);

    public static final UnaryGradientFunction RELU_GRAD = (l, o, oGrad) -> {
        double temp = l;
        temp = (temp == -0.0) ? 0.0 : temp;
        return oGrad * ((temp >= 0) ? 1 : 0);
    };
    public static final UnaryGradientFunction LEAKY_RELU_GRAD = (l, o, oGrad) -> {
        double temp = l;
        temp = (temp == -0.0) ? 0.0 : temp;
        return oGrad * ((temp >= 0) ? 1 : 0.1);
    };
    public static final UnaryGradientFunction EXP_GRAD = (l, o, oGrad) -> oGrad * o;
    public static final UnaryGradientFunction LN_GRAD = (l, o, oGrad) -> oGrad * (1 / nanGuard(l));

    private static final double LN10 = Math.log(10);

    public static final UnaryGradientFunction LOG_GRAD = (l, o, oGrad) -> oGrad * (1 / (LN10 * nanGuard(l)));
    public static final UnaryGradientFunction SIGMOID_GRAD = (l, o, oGrad) -> {
        double sigmoid = o;
        double dSigmoid = sigmoid * (1 - sigmoid);
        return oGrad * dSigmoid;
    };
    public static final UnaryGradientFunction SIMPLE_SWISH_GRAD = (l, o, oGrad) -> {
        double x = l;
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));
        double swishGrad = sigmoid + x * sigmoid * (1.0 - sigmoid);
        return oGrad * swishGrad;
    };

    @FunctionalInterface
    public static interface UnaryCalcFunction {
        public double result(double input);
    }
    @FunctionalInterface
    public static interface BinaryCalcFunction {
        public double result(double left, double right);
    }
    @FunctionalInterface
    public static interface UnaryGradientFunction {
        public double result(double input, double result, double grad);
    }
    @FunctionalInterface
    public static interface BinaryGradientFunction {
        public double result(double left, double right, double result, double grad);
    }
}
