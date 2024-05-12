package com.ashutoshwad.utils.jautograd.stepdefinitions;

import static org.junit.Assert.assertEquals;

import com.ashutoshwad.utils.jautograd.Value;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

public class NeuronStepDefinition {
	private Value i1;
	private Value i2;
	private Value w1;
	private Value w2;
	private Value b1;
	private Value o1;
	private Value e1;
	private Value loss;
	@Given("I create a neuron with inputs i1 & i2, bias b1 and output o1")
	public void i_create_a_neuron_with_inputs_i1_i2_bias_b1_and_output_o1() {
		i1 = Value.of(0.2);
		i2 = Value.of(0.3);
		w1 = Value.of(0.4);
		w2 = Value.of(0.5);
		b1 = Value.of(0.6);
		o1 = i1.mul(w1).add(i2.mul(w2)).add(b1).tanh();
		e1 = Value.of(0.1);
	}
	@Given("define loss l1 as \\(o1)^2")
	public void define_loss_l1_as_o1() {
	    loss = o1.sub(e1);
	    loss = loss.mul(loss);
	}
	@When("I complete the forward and backward pass")
	public void i_complete_the_forward_and_backward_pass() {
		loss.forward();
		loss.backward();
	}
	@Then("The gradients must be correct as calculated by the expression \\(f\\(x+h)-f\\(x))\\/h")
	public void the_gradients_must_be_correct_as_calculated_by_the_expression_f_x_h_f_x_h() {
		double fx = loss.getValue();
		double h = 0.0001;
		i1.setValue(i1.getValue() + h);
		loss.forward();
		double fxPlush = loss.getValue();
		double grad = (fxPlush - fx) / h;
		assertEquals(i1.getGradient(), grad, 0.00001);
		i1.setValue(0.2 - grad);
		loss.forward();
	}
}
