package com.ashutoshwad.utils.jautograd.stepdefinitions;

import static org.junit.Assert.assertEquals;

import com.ashutoshwad.utils.jautograd.JAutogradValue;
import com.ashutoshwad.utils.jautograd.Value;

import io.cucumber.java.Before;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

import java.util.LinkedList;
import java.util.List;

public class CucumberStepDefinitions {
	private Value x;
	private Value y;

	private Value x_array[];
	private Value x_array_result[];
	private double x_array_gradient[];

	private double deviation;
	private double h;
	private Value result;
	private double xGradient;
	private double yGradient;

	@Before
	public void initialize() {
		x = null;
		y = null;
		result = null;
		deviation = 0;
		h = 0;
		xGradient = 0;
		yGradient = 0;
	}

	@Given("I initialize variable x to value {double}")
	public void i_initialize_variable_x_to_value(Double x) {
	    this.x = new JAutogradValue(x);
	}

	@Given("I initialize y to value {double}")
	public void i_initialize_y_to_value(Double y) {
		this.y = new JAutogradValue(y);
	}

	@Given("allowed deviation is {double}")
	public void and_allowed_deviation_is(Double deviation) {
		this.deviation = deviation;
	}

	@Then("the result should be {double}")
	public void the_result_should_be_with_deviation_allowed(Double result) {
	    assertEquals(result, this.result.getValue(), deviation);
	}
	@Then("I must see the result equal to {double}")
	public void i_must_see_the_result_equal_to(Double result) {
		assertEquals(result, this.result.getValue(), deviation);
	}
	@Then("the gradient of x should be as calculated")
	public void the_gradient_of_x_should_be() {
		assertEquals(xGradient, x.getGradient(), deviation);
	}

	@Then("the gradient of y should be as calculated")
	public void the_gradient_of_y_should_be() {
		assertEquals(yGradient, y.getGradient(), deviation);
	}

	@Given("h is {double}")
	public void h_is(Double h) {
	    this.h = h;
	}

	/*------------------- Operators start here ---------------------------*/
	@When("I perform operation ADDITION on x and y")
	public void i_perform_operation_addition_on_x_and_y() {
	    result = x.add(y);
	    result.backward();
	    
	    double xTemp = x.getValue();
	    double yTemp = y.getValue();
	    //Calculate gradient for x
	    double nudgeF = (xTemp + h) + yTemp;
	    double f = result.getValue();
	    xGradient = (nudgeF - f) / h;
	    //Calculate gradient for y
	    nudgeF = xTemp + (yTemp + h);
	    f = result.getValue();
	    yGradient = (nudgeF - f) / h;
	}

	@When("I perform operation SUBTRACTION on x and y")
	public void i_perform_operation_subtract_on_x_and_y() {
		result = x.sub(y);
	    result.backward();
	    
	    double xTemp = x.getValue();
	    double yTemp = y.getValue();
	    //Calculate gradient for x
	    double nudgeF = (xTemp + h) - yTemp;
	    double f = result.getValue();
	    xGradient = (nudgeF - f) / h;
	    //Calculate gradient for y
	    nudgeF = xTemp - (yTemp + h);
	    f = result.getValue();
	    yGradient = (nudgeF - f) / h;
	}

	@When("I perform operation MULTIPLICATION on x and y")
	public void i_perform_operation_multiplication_on_x_and_y() {
		result = x.mul(y);
	    result.backward();
	    
	    double xTemp = x.getValue();
	    double yTemp = y.getValue();
	    //Calculate gradient for x
	    double nudgeF = (xTemp + h) * yTemp;
	    double f = result.getValue();
	    xGradient = (nudgeF - f) / h;
	    //Calculate gradient for y
	    nudgeF = xTemp * (yTemp + h);
	    f = result.getValue();
	    yGradient = (nudgeF - f) / h;
	}

	@When("I perform operation DIVISION on x and y")
	public void i_perform_operation_division_on_x_and_y() {
		result = x.div(y);
	    result.backward();
	    
	    double xTemp = x.getValue();
	    double yTemp = y.getValue();
	    //Calculate gradient for x
	    double nudgeF = (xTemp + h) / yTemp;
	    double f = result.getValue();
	    xGradient = (nudgeF - f) / h;
	    //Calculate gradient for y
	    nudgeF = xTemp / (yTemp + h);
	    f = result.getValue();
	    yGradient = (nudgeF - f) / h;
	}

	@When("I perform operation EXPONENTIATION on x and y")
	public void i_perform_operation_exponentiation_on_x_and_y() {
		result = x.pow(y);
	    result.backward();
	    
	    double xTemp = x.getValue();
	    double yTemp = y.getValue();
	    //Calculate gradient for x
	    double nudgeF = Math.pow((xTemp + h), yTemp);
	    double f = result.getValue();
	    xGradient = (nudgeF - f) / h;
	    //Calculate gradient for y
	    nudgeF = Math.pow(xTemp, (yTemp + h));
	    f = result.getValue();
	    yGradient = (nudgeF - f) / h;
	}
	/*------------------- Functions start here ---------------------------*/
	@When("I invoke the function RELU")
	public void i_invoke_the_function_relu() {
		result = x.relu();
		result.backward();
		if(x.getValue()>=0) {
			xGradient = 1;
		} else {
			xGradient = 0;
		}
	}
	@When("I invoke the function NEG_RELU_0_7")
	public void i_invoke_the_function_neg_relu_0_7() {
		result = x.relu(0.7);
		result.backward();
		if(x.getValue()>=0) {
			xGradient = 1;
		} else {
			xGradient = 0.7;
		}
	}

	@Given("I initialize variable x_array to a range of values between {int} and {int} with a step size of {double}")
	public void i_initialize_variable_x_array_to_a_range_of_values_between_and_with_a_step_size_of(double start, double end, double step) {
		List<Value> values = new LinkedList<>();
		for (double i = start; i <= end; i+=step) {
			values.add(new JAutogradValue(i));
		}

		x_array = new Value[values.size()];
		int i = 0;
		for (Value val : values) {
			x_array[i++] = val;
		}
	}
	@When("I invoke the function SIN on x_array")
	public void i_invoke_the_function_sin_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].sin();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.sin(xTemp + h);
			double f = Math.sin(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for SIN")
	public void i_must_see_the_result_equal_to_the_expected_result_for_sin() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.sin(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function COS on x_array")
	public void i_invoke_the_function_cos_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].cos();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.cos(xTemp + h);
			double f = Math.cos(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for COS")
	public void i_must_see_the_result_equal_to_the_expected_result_for_cos() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.cos(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function TAN on x_array")
	public void i_invoke_the_function_tan_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].tan();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.tan(xTemp + h);
			double f = Math.tan(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for TAN")
	public void i_must_see_the_result_equal_to_the_expected_result_for_tan() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.tan(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function SINH on x_array")
	public void i_invoke_the_function_sinh_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].sinh();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.sinh(xTemp + h);
			double f = Math.sinh(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for SINH")
	public void i_must_see_the_result_equal_to_the_expected_result_for_sinh() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.sinh(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function COSH on x_array")
	public void i_invoke_the_function_cosh_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].cosh();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.cosh(xTemp + h);
			double f = Math.cosh(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for COSH")
	public void i_must_see_the_result_equal_to_the_expected_result_for_cosh() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.cosh(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function TANH on x_array")
	public void i_invoke_the_function_tanh_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].tanh();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.tanh(xTemp + h);
			double f = Math.tanh(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for TANH")
	public void i_must_see_the_result_equal_to_the_expected_result_for_tanh() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.tanh(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function EXPONENTIAL on x_array")
	public void i_invoke_the_function_exponential_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].exponential();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = Math.exp(xTemp + h);
			double f = Math.exp(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for EXPONENTIAL")
	public void i_must_see_the_result_equal_to_the_expected_result_for_exponential() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(Math.exp(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}
	@When("I invoke the function SIGMOID on x_array")
	public void i_invoke_the_function_sigmoid_on_x_array() {
		x_array_result = new Value[x_array.length];
		x_array_gradient = new double[x_array.length];
		for (int i = 0; i < x_array.length; i++) {
			x_array_result[i] = x_array[i].sigmoid();
			x_array_result[i].backward();


			double xTemp = x_array[i].getValue();
			//Calculate gradient for x
			double nudgeF = sigmoid(xTemp + h);
			double f = sigmoid(xTemp);
			x_array_gradient[i] = (nudgeF - f) / h;
		}
	}
	@Then("I must see the result equal to the expected result for SIGMOID")
	public void i_must_see_the_result_equal_to_the_expected_result_for_sigmoid() {
		for (int i = 0; i < x_array_result.length; i++) {
			assertEquals(sigmoid(x_array[i].getValue()), x_array_result[i].getValue(), deviation);
		}
	}

	private double sigmoid(double value) {
		double eRaisedToX = Math.exp(value);
		return eRaisedToX / (1 + eRaisedToX);
	}

	@Then("the gradient of x_array should be as calculated")
	public void the_gradient_of_x_array_should_be_as_calculated() {
		for (int i = 0; i < x_array_gradient.length; i++) {
			assertEquals(x_array_gradient[i], x_array[i].getGradient(), deviation);
		}
	}

	@Then("the gradient of x_array should be as calculated for TAN")
	public void the_gradient_of_x_array_should_be_as_calculated_for_tan() {
		for (int i = 0; i < x_array_gradient.length; i++) {
			double expected = makeSmall(x_array_gradient[i]);
			double actual = makeSmall(x_array[i].getGradient());

			assertEquals(expected, actual, deviation);
		}
	}
	private double makeSmall(double value) {
		while(value>1) {
			value = value/10;
		}
		return value;
	}
}
