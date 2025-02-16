package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.*;

public class BytePairEncoder {
    private static final Logger log = LoggerFactory.getLogger(BytePairEncoder.class);

    public static void buildEncoding(File input, final long maxTokens) {
        String corpus = readFileIntoString(input);
        List<Token> tokens = extractBaseTokens(corpus);

        Map<Long, Token>tokenLookupMap = new HashMap<>();
        Map<String, Token>idLookupMap = new HashMap<>();
        buildLookupMaps(tokens, tokenLookupMap, idLookupMap);

        List<Token>corpusAsTokens = new LinkedList<>();
        for (int i = 0; i < corpus.length(); i++) {
            corpusAsTokens.add(idLookupMap.get(Character.toString(corpus.charAt(i))));
        }
        long idProvider = tokens.get(tokens.size() - 1).getId() + 1;

        while(tokens.size() < maxTokens && corpusAsTokens.size()>1) {
            TokenPair mostCommonPair = findMaxFrequencyPair(corpusAsTokens);
            Token newToken = new Token(idProvider++, mostCommonPair.getOne().getToken() + mostCommonPair.getTwo().getToken());
            tokens.add(newToken);
            corpusAsTokens = replaceInCorpus(mostCommonPair, newToken, corpusAsTokens);
            printByProbability(0.1, tokens);
        }
        for(Token token:tokens) {
            System.out.println(token.getId() + ": \"" + token + "\"");
        }
    }

    public static void printByProbability(double probability, List<Token> tokens) {
        // Ensure probability is within [0, 1] range
        double finalProbability = Math.max(0.0, Math.min(1.0, probability));
        Random r = new Random();

        if (r.nextDouble() < finalProbability) {
            for(Token token:tokens) {
                System.out.println(token.getId() + ": \"" + token.getToken() + "\"");
            }
        }
    }

    private static List<Token> replaceInCorpus(TokenPair mostCommonPair, Token mergedToken, List<Token> corpusAsTokens) {
        List<Token>newCorpus = new LinkedList<>();
        boolean MATCH = false;
        Token old = null;
        for(Token token:corpusAsTokens) {
            if(MATCH) {
                if(null == old || null == mergedToken || null == token) {
                    System.out.println("asd");
                }
                if(token.getId() == mostCommonPair.getTwo().getId()) {
                    MATCH=false;
                    newCorpus.add(mergedToken);
                    old = null;
                } else if(token.getId() == mostCommonPair.getOne().getId()) {
                    MATCH = true;
                    newCorpus.add(old);
                    old = token;
                } else {
                    MATCH=false;
                    newCorpus.add(old);
                    newCorpus.add(token);
                    old = null;
                }
            } else {
                if(token.getId() == mostCommonPair.getOne().getId()) {
                    MATCH = true;
                    old = token;
                } else {
                    newCorpus.add(token);
                }
            }
        }
        boolean activate = false;
        int count = 0;
        int i=0;

        for(Token token:newCorpus ) {
            if(null == token) {
                activate = true;
                count = 1000;
                System.out.println(i);
            }
            if (activate) {
                if(count>0) {
                    System.out.print(token);
                    count--;
                } else {
                    activate = false;
                }
            }
            i++;
        }
        return newCorpus;
    }

    private static TokenPair findMaxFrequencyPair(List<Token> corpusAsTokens) {
        Map<TokenPair, Long>frequencyTrackingMap = new HashMap<>();
        Token old = null;
        long start = System.currentTimeMillis();
        System.out.println("Start loop for calculating max frequency pair");
        for (Token token : corpusAsTokens) {
            if(null == old) {
                old = token;
            } else {
                TokenPair tokenPair = new TokenPair(old, token);
                if(!frequencyTrackingMap.containsKey(tokenPair)) {
                    frequencyTrackingMap.put(tokenPair, 1l);
                } else {
                    frequencyTrackingMap.put(tokenPair, frequencyTrackingMap.get(tokenPair)+1);
                }
                old = token;
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("Took " + (end-start) + " milliseconds for one run.");
        long max = 0;
        TokenPair selectedKey = null;
        for (Map.Entry<TokenPair, Long>entry:frequencyTrackingMap.entrySet()) {
            if(entry.getValue()>max) {
                selectedKey = entry.getKey();
            }
        }
        return selectedKey;
    }

    private static void buildLookupMaps(List<Token> tokens, Map<Long, Token> tokenLookupMap, Map<String, Token> idLookupMap) {
        for(Token token:tokens) {
            tokenLookupMap.put(token.getId(), token);
            idLookupMap.put(token.getToken(), token);
        }
    }

    private void voidPrintTokens(List<Token> tokens) {
        for (Token token:tokens) {
            System.out.println("\t"+token.getId() + ":\t"+token.getToken());
        }
    }

    private static List<Token> extractBaseTokens(String corpus) {
        Set<String>tokenSet = new LinkedHashSet<>();
        for (int i = 0; i < corpus.length(); i++) {
            tokenSet.add(Character.toString(corpus.charAt(i)));
        }

        long id = 0;
        List<Token>modelList = new LinkedList<>();
        modelList.add(new Token(id++, null));

        for (String token:tokenSet) {
            modelList.add(new Token(id++, token));
        }

        return modelList;
    }

    private static String readFileIntoString(File input) {
        StringBuilder sb = new StringBuilder((int)input.length());
        try {
            Scanner scn = new Scanner(new BufferedInputStream(new FileInputStream(input)));
            while(scn.hasNextLine()) {
                sb.append(preProcessInput(scn.nextLine()));
                sb.append("\n");
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        return sb.toString();
    }

    private static String preProcessInput(String str) {
        //Remove any occurance of multiple spaces
        while(str.contains("  ")) {
            str = str.replace("  ", " ");
        }
        return str;
    }
}
