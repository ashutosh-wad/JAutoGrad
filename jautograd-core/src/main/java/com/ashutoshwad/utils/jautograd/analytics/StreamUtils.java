package com.ashutoshwad.utils.jautograd.analytics;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class StreamUtils {
    public static Stream<Map<String, String>> getCsvStream(String fileName) {
        File inFile = new File(fileName);
        return StreamSupport.stream(new CsvSpliterator(inFile), false);
    }
    private static class CsvSpliterator implements Spliterator<Map<String, String>> {
        private Scanner scn;
        private List<String> headers;
        public CsvSpliterator(File csvFile) {
            try {
                scn = new Scanner(csvFile);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
            if(!scn.hasNextLine()) {
                throw new RuntimeException("CSV files must have a header");
            }
            headers = csvSplit(nextRow());
        }

        private boolean isOpen() {
            if(null == scn) {
                return false;
            }
            return scn.hasNextLine();
        }

        private String nextRow() {
            if(scn.hasNextLine()) {
                return scn.nextLine();
            } else {
                scn.close();
                scn = null;
                return null;
            }
        }

        private List<String> csvSplit(String row) {
            List<String>csvList = new ArrayList<>();
            StringBuilder sb = new StringBuilder();
            char[]ch = row.toCharArray();
            boolean inString = false;
            int quote = 0;
            for (int i = 0; i < ch.length; i++) {
                if(!inString) {
                    if('\'' == ch[i]) {
                        inString = true;
                        quote = 1;
                    } else if('\"' == ch[i]) {
                        inString = true;
                        quote = 2;
                    } else if(',' == ch[i]) {
                        csvList.add(sb.toString());
                        sb = new StringBuilder();
                    } else {
                        sb.append(ch[i]);
                    }
                } else {
                    if('\'' == ch[i] && 1 == quote) {
                        inString = false;
                        quote = 0;
                    } else if('\"' == ch[i] && 2 == quote) {
                        inString = false;
                        quote = 0;
                    } else {
                        sb.append(ch[i]);
                    }
                }
            }
            csvList.add(sb.toString());
            return csvList;
        }

        @Override
        public boolean tryAdvance(Consumer<? super Map<String, String>> action) {
            if(!isOpen()) {
                return false;
            }
            List<String>row = csvSplit(nextRow());
            Map<String, String>result = new HashMap<>();
            for (int i = 0; i < row.size(); i++) {
                if(i<headers.size()) {
                    result.put(headers.get(i), row.get(i));
                } else {
                    result.put("$COLUMN_"+i, row.get(i));
                }
            }
            action.accept(result);
            return true;
        }

        @Override
        public Spliterator<Map<String, String>> trySplit() {
            return null;
        }

        @Override
        public long estimateSize() {
            return isOpen()?1:0;
        }

        @Override
        public int characteristics() {
            return ORDERED | NONNULL;
        }
    }
}
