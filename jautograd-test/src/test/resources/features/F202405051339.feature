Feature: Arithmatic operators supported by JAutogradValue

  Scenario: JAutogradValue must support all standard arithmatic operators
    Given I initialize variable x to value <X>
    And I initialize y to value <Y>
    And allowed deviation is 0.00001
    And h is 0.000001
    When I perform operation <OP> on x and y
    Then the result should be <result>
    And the gradient of x should be as calculated
    And the gradient of y should be as calculated
  Examples:
		| X  | Y  | OP             | result        |
		| 1  | 1  | EXPONENTIATION | 1             |
		| 1  | -1 | EXPONENTIATION | 1             |
		| -1 | 1  | EXPONENTIATION | -1            |
		| -1 | -1 | EXPONENTIATION | -1            |
		| 2  | 3  | EXPONENTIATION | 8             |
		| 2  | -3 | EXPONENTIATION | 0.125         |
		| -2 | 3  | EXPONENTIATION | -8            |
		| -2 | -3 | EXPONENTIATION | -0.125        |
		| 1  | 1  | ADDITION       | 2             |
		| 1  | -1 | ADDITION       | 0             |
		| -1 | 1  | ADDITION       | 0             |
		| -1 | -1 | ADDITION       | -2            |
		| 2  | 3  | ADDITION       | 5             |
		| 2  | -3 | ADDITION       | -1            |
		| -2 | 3  | ADDITION       | 1             |
		| -2 | -3 | ADDITION       | -5            |
		| 1  | 1  | SUBTRACTION    | 0             |
		| 1  | -1 | SUBTRACTION    | 2             |
		| -1 | 1  | SUBTRACTION    | -2            |
		| -1 | -1 | SUBTRACTION    | 0             |
		| 2  | 3  | SUBTRACTION    | -1            |
		| 2  | -3 | SUBTRACTION    | 5             |
		| -2 | 3  | SUBTRACTION    | -5            |
		| -2 | -3 | SUBTRACTION    | 1             |
		| 1  | 1  | MULTIPLICATION | 1             |
		| 1  | -1 | MULTIPLICATION | -1            |
		| -1 | 1  | MULTIPLICATION | -1            |
		| -1 | -1 | MULTIPLICATION | 1             |
		| 2  | 3  | MULTIPLICATION | 6             |
		| 2  | -3 | MULTIPLICATION | -6            |
		| -2 | 3  | MULTIPLICATION | -6            |
		| -2 | -3 | MULTIPLICATION | 6             |
		| 1  | 1  | DIVISION       | 1             |
		| 1  | -1 | DIVISION       | -1            |
		| -1 | 1  | DIVISION       | -1            |
		| -1 | -1 | DIVISION       | 1             |
		| 2  | 3  | DIVISION       | 0.6666666666  |
		| 2  | -3 | DIVISION       | -0.6666666666 |
		| -2 | 3  | DIVISION       | -0.6666666666 |
		| -2 | -3 | DIVISION       | 0.6666666666  |


