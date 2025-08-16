package com.ashutoshwad.utils.jautograd;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ExecutorFactory {
    private static ExecutorService executorService;
    private static int numThreads;

    static {
        synchronized (ExecutorFactory.class) {
            ExecutorFactory.numThreads = Runtime.getRuntime().availableProcessors();
            ExecutorFactory.executorService = Executors.newFixedThreadPool(numThreads);
        }
    }

    public static void createExecutor(int numThreads) {
        synchronized (ExecutorFactory.class) {
            if (executorService != null) {
                executorService.shutdown();
            }
            numThreads = Math.max(1, numThreads);
            executorService = Executors.newFixedThreadPool(numThreads);
        }
    }

    public record Details(ExecutorService executorService, int numThreads){}
    public static Details getDetails() {
        synchronized (ExecutorFactory.class) {
            return new Details(executorService, numThreads);
        }
    }
}
