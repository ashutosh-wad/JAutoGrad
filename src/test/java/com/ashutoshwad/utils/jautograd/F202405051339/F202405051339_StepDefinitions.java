package com.ashutoshwad.utils.jautograd.F202405051339;

import static org.junit.Assert.assertEquals;

import com.ashutoshwad.utils.jautograd.JAutogradValue;
import com.ashutoshwad.utils.jautograd.Value;

import io.cucumber.java.Before;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

public class F202405051339_StepDefinitions {
	private Value x;
	private Value y;
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

	/*-------------------Operators start here ---------------------------*/
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
}
