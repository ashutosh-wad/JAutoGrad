package com.ashutoshwad.utils.jautograd;

import org.junit.runner.RunWith;

import io.cucumber.junit.Cucumber;
import io.cucumber.junit.CucumberOptions;

@RunWith(Cucumber.class)
@CucumberOptions(
		features = "src/test/resources/features/F202405051339.feature",
		glue = "com.ashutoshwad.utils.jautograd.F202405051339",
		plugin = {"pretty", "html:cucumber_report.html"},
		dryRun = false
		)
public class TestCaseRunner {
}
