Feature: Functions supported by JAutogradValue should work as expected.

  Scenario: JAutogradValue must support RELU for calculations.
    Given I initialize variable x to value <X>
    And allowed deviation is 0.00001
    And h is 0.000001
    When I invoke the function <function>
    Then I must see the result equal to <result>
    And the gradient of x should be as calculated
    Examples:
      | X      | function     | result  |
      | 1      | RELU         | 1       |
      | 0      | RELU         | 0       |
      | 0.001  | RELU         | 0.001   |
      | -0.001 | RELU         | 0       |
      | -1     | RELU         | 0       |
      | 1      | NEG_RELU_0_7 | 1       |
      | 0      | NEG_RELU_0_7 | 0       |
      | 0.001  | NEG_RELU_0_7 | 0.001   |
      | -0.001 | NEG_RELU_0_7 | -0.0007 |
      | -1     | NEG_RELU_0_7 | -0.7    |

  Scenario: JAutogradValue must support trigonometric functions for calculations.
    Given I initialize variable x_array to a range of values between -2 and 2 with a step size of 0.01
    And allowed deviation is 0.00001
    And h is 0.00000001
    When I invoke the function <function> on x_array
    Then I must see the result equal to the expected result for <function>
    And the gradient of x_array should be as calculated
    Examples:
      | function    |
      | SIN         |
      | COS         |
      | SINH        |
      | COSH        |
      | TANH        |
      | SIGMOID     |
      | EXPONENTIAL |

  Scenario: JAutogradValue must support trigonometric functions for calculations.
    Given I initialize variable x_array to a range of values between -2 and 2 with a step size of 0.01
    And allowed deviation is 0.0001
    And h is 0.00000001
    When I invoke the function TAN on x_array
    Then I must see the result equal to the expected result for TAN
    And the gradient of x_array should be as calculated for TAN
