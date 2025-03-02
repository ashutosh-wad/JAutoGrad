# Jautograd compute

A lot has been acomplished, let me list out a checklist of pending tasks.

- [ ] Create a parallel executor that can execute the compute graph forward and backward in a parallel wat
- [ ] Reorganize and design code from the perspective of the end user
- [ ] Complete all javadocs and create documentation on how to use this repository
- [ ] Implement logging with slf4j and logback (perhaps logback is not needed, depends on analysis)
- [ ] Validate and correct maven dependencies across pom's.
  - Other than logging and testing, no external libraries should be used if it can be helped
- [ ] Complete all JUnit test cases
- [ ] Target PItest coverage to 100%
- [ ] Ensure all old JAutograd test cases pass
- [ ] Update main readme to have all details to build project from scratch as well as implement any neural network
- [ ] Implement test project for mnist number recognition dataset
- [ ] Use names.txt from Andrej Karpathy's video to implement a name generator
  - Train, Save and load this model successfully
  - Make it a permanent class
- [ ] Deploy release version to maven central