package com.ashutoshwad.utils.jautograd;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class ExecutorUtils {
    public static void awaitFutures(List<Future<?>> futures) {
        for (Future<?>future:futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Worker was interrupted", e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Worker failed to execute", e);
            }
        }
    }
}
