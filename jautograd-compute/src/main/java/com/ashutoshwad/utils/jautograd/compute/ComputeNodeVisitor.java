package com.ashutoshwad.utils.jautograd.compute;

import java.util.*;

public class ComputeNodeVisitor {
    private Queue<ComputeNode> visitQueue;
    private Map<Long, Set<Long>> dependencyMap;
    private Map<Long, Set<Long>> reverseDependencyMap;
    private Map<Long, ComputeNode> nodeMap;
    private List<ComputeNode[]> batches;

    public ComputeNodeVisitor() {
        init();
    }

    public void init() {
        this.visitQueue = new LinkedList<>();
        this.dependencyMap = new HashMap<>();
        this.reverseDependencyMap = new HashMap<>();
        this.nodeMap = new HashMap<>();
        this.batches = new LinkedList<>();
    }

    private void addDependency(long source, long target) {
        if (!dependencyMap.containsKey(source)) {
            dependencyMap.put(source, new HashSet<>());
        }
        if (!reverseDependencyMap.containsKey(target)) {
            reverseDependencyMap.put(target, new HashSet<>());
        }
        dependencyMap.get(source).add(target);
        reverseDependencyMap.get(target).add(source);
    }

    private void enqueueVisit(ComputeNode node) {
        if (null == node) {
            return;
        }
        if (nodeMap.containsKey(node.getId())) {
            return;
        }
        visitQueue.add(node);
    }

    private void visitQueuedNodes() {
        while (!visitQueue.isEmpty()) {
            ComputeNode node = visitQueue.poll();
            node.visit(this);
        }
    }

    public void visit(ComputeNode src, ComputeNode left, ComputeNode right, ComputeNode[] dependencies) {
        nodeMap.put(src.getId(), src);

        if (left != null) {
            addDependency(src.getId(), left.getId());
        }
        if (right != null) {
            addDependency(src.getId(), right.getId());
        }

        enqueueVisit(src);
        enqueueVisit(left);
        enqueueVisit(right);
        if (null != dependencies) {
            for (int i = 0; i < dependencies.length; i++) {
                enqueueVisit(dependencies[i]);
            }
        }
        visitQueuedNodes();
    }

    public void prepareBatches() {
        while (!nodeMap.isEmpty()) {
            ComputeNode[] batch = prepareBatch();
            batches.add(batch);
        }
    }

    public ComputeNode[] prepareBatch() {
        List<ComputeNode> batch = new LinkedList<>();
        /*
        for (Map.Entry<Long, ComputeNode> entry:nodeMap.entrySet()) {
            if (!dependencyMap.containsKey(entry.getKey())) {
                batch.add(entry.getValue());
            }
        }
        for (Iterator<Map.Entry<Long, ComputeNode>> it = nodeMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<Long, ComputeNode> entry = it.next();
            Long key = entry.getKey();
            ComputeNode node = entry.getValue();

            if (!dependencyMap.containsKey(key)) {
                it.remove();
                if (reverseDependencyMap.containsKey(key)) {
                    for (long depKey : reverseDependencyMap.get(key)) {
                        Set<Long> dependencySet = dependencyMap.get(depKey);
                        dependencySet.remove(key);
                        if (dependencySet.isEmpty()) {
                            dependencyMap.remove(depKey);
                        }
                    }
                    reverseDependencyMap.remove(key);
                }
            }
        }
        */
        for (Iterator<Map.Entry<Long, ComputeNode>> it = nodeMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<Long, ComputeNode> entry = it.next();
            Long key = entry.getKey();
            ComputeNode node = entry.getValue();
            if (!dependencyMap.containsKey(key)) {
                batch.add(node);
                it.remove();
            }
        }
        for(ComputeNode node:batch) {
            Long key = node.getId();
            if (reverseDependencyMap.containsKey(key)) {
                for (long depKey : reverseDependencyMap.get(key)) {
                    Set<Long> dependencySet = dependencyMap.get(depKey);
                    dependencySet.remove(key);
                    if (dependencySet.isEmpty()) {
                        dependencyMap.remove(depKey);
                    }
                }
                reverseDependencyMap.remove(key);
            }
        }

        Iterator<ComputeNode> it = batch.iterator();
        ComputeNode[] batchArr = new ComputeNode[batch.size()];
        for (int i = 0; i < batchArr.length; i++) {
            batchArr[i] = it.next();
        }
        return batchArr;
    }
}
