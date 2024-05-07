package com.ashutoshwad.utils.jautograd.stepdefinitions;

import static org.junit.Assert.assertEquals;

import java.util.LinkedList;
import java.util.List;

import com.ashutoshwad.utils.jautograd.JAutogradValue;
import com.ashutoshwad.utils.jautograd.Value;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

public class TanhStepDefinition {
	private Value x;
	private Value tanh;
	private List<Double>input;
	private List<Double>output;
	private List<Double>mathTanhOutput;
	private List<Double>gradient;
	@Given("I create a tanh implementation using JautoGradValue")
	public void i_create_a_tanh_implementation_using_jauto_grad_value() {
		x = new JAutogradValue(0);
	    Value twoX = new JAutogradValue(2).mul(x);
	    Value eToX = twoX.exponential();
	    Value one = new JAutogradValue(1);
	    tanh = eToX.sub(one).div(eToX.add(one));
	}
	@When("I invoke the tanh function i created on a range of values between {double} and {double} with step {double}")
	public void i_invoke_the_tanh_function_i_created_on_a_range_of_values_between_and_with_step(double start, double end, double step) {
		input = new LinkedList<Double>();
		for(double i = start;i<=end;i+=step) {
			input.add(i);
		}
		output = new LinkedList<Double>();
		gradient = new LinkedList<Double>();
		mathTanhOutput = new LinkedList<Double>();
		for(Double i:input) {
			x.setValue(i);
			tanh.forward();
			output.add(tanh.getValue());
			tanh.backward();
			gradient.add(x.getGradient());
			tanh.reset();
			mathTanhOutput.add(Math.tanh(i));
		}
	}
	@Then("all values match with Math.tanh")
	public void all_values_match_with_math_tanh() {
	    for (int i = 0; i < output.size(); i++) {
			assertEquals(mathTanhOutput.get(i), output.get(i), 0.0000001);
		}
	}
	@Then("all gradients match with 1\\/cosh^2\\(x)")
	public void all_gradients_match_with_cosh_x() {
		for (int i = 0; i < output.size(); i++) {
			assertEquals(1/Math.pow(Math.cosh(input.get(i)), 2), gradient.get(i), 0.0000001);
		}
	}
}
