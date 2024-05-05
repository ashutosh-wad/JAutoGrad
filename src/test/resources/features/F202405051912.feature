@Functions
Feature: Functions supported by JAutogradValue should work as expected.

  Scenario: JAutogradValue must support standard functions for calculations.
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