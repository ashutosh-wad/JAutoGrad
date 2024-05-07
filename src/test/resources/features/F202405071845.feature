Feature: A neuron modeled for a neural network must have gradients accurately propogated,

  Scenario: A neuron with 2 inputs(i1, i2), bias (b1) and one output(o1) is created, gradients are calculated for the input and bias with a standard expression. The gradients calculated by autograd must be correct.
    Given I create a neuron with inputs i1 & i2, bias b1 and output o1
    And define loss l1 as (o1)^2
    When I complete the forward and backward pass
    Then The gradients must be correct as calculated by the expression (f(x+h)-f(x))/h

	Scenario: An implementation of tanh must calculate the correct result and gradient
		Given I create a tanh implementation using JautoGradValue
		When I invoke the tanh function i created on a range of values between -1 and 1 with step 0.1
		Then all values match with Math.tanh
		And all gradients match with 1/cosh^2(x)