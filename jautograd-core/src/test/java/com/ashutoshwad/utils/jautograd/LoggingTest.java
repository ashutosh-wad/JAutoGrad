package com.ashutoshwad.utils.jautograd;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LoggingTest {
    private static final Logger log = LoggerFactory.getLogger(LoggingTest.class);

    @Test
    public void loggingTest() {
        log.info("Testing");
    }
}
