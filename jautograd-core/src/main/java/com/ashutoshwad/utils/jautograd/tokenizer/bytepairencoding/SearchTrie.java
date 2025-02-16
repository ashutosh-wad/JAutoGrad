package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import java.util.HashMap;
import java.util.Map;

public class SearchTrie {
    private Node root;

    public SearchTrie() {
        root = new Node();
    }

    public void clear() {
        root = new Node();
    }

    public void add(String str, Long id) {
        if(str==null || str.isEmpty()) {
            return;
        }
        Node terminalNode = root;
        terminalNode.add(str, id);
    }

    public QueryResponse query(String str) {
        Node result = root.find(str);
        if(null==result) {
            return new QueryResponse(false, -1);
        }
        return new QueryResponse(true, result.id);
    }

    public static class QueryResponse {
        private final boolean path;
        private final long id;

        public QueryResponse(boolean path, long id) {
            this.path = path;
            this.id = id;
        }

        public boolean hasPath() {
            return path;
        }

        public long getId() {
            return id;
        }
    }

    private static class Node {
        private Map<Character, Node> referenceMap;
        private long id;

        public Node() {
            id = -1;
            referenceMap = new HashMap<>();
        }

        public boolean hasPath(char ch) {
            return referenceMap.containsKey(ch);
        }

        public long getId() {
            return id;
        }

        public void setId(long id) {
            this.id = id;
        }

        public void add(String str, long id) {
            add(0, str, id);
        }

        private void add(int index, String str, long id) {
            referenceMap.computeIfAbsent(str.charAt(index), (k)->new Node());
            int nextIndex = index+1;
            if(nextIndex < str.length()) {
                referenceMap.get(str.charAt(index)).add(nextIndex, str, id);
            } else {
                referenceMap.get(str.charAt(index)).setId(id);
            }
        }

        public Node find(String str) {
            return find(0, str);
        }

        private Node find(int index, String str) {
            if(referenceMap.containsKey(str.charAt(index))) {
                if(index==str.length()-1) {
                    return referenceMap.get(str.charAt(index));
                } else {
                    return referenceMap.get(str.charAt(index)).find(index+1, str);
                }
            } else {
                return null;
            }
        }
    }
}
