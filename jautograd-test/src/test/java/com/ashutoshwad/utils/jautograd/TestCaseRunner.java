package com.ashutoshwad.utils.jautograd;

import org.junit.runner.RunWith;

import io.cucumber.junit.Cucumber;
import io.cucumber.junit.CucumberOptions;

@RunWith(Cucumber.class)
@CucumberOptions(
		features = "src/test/resources/features",
		glue = "com.ashutoshwad.utils.jautograd.stepdefinitions",
		plugin = {"html:cucumber_report.html"},
		dryRun = false
		)
public class TestCaseRunner {
}
