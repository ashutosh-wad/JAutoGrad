package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class BytePairTokenizer {
    private final static String TOKEN_FILE_LOCATION = "<whatever file>";
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private final String fileName;
    private Set<Token> tokenSet;
    private Map<Long, Token> tokenIndexById;
    private SearchTrie tokenTrie;
    private Token END_OF_INPUT;
    private long nextId;

    public BytePairTokenizer() {
        this(TOKEN_FILE_LOCATION);
    }

    public BytePairTokenizer(String fileName) {
        this.fileName = fileName;
        this.tokenSet = new HashSet<>();
        this.tokenIndexById = new HashMap<>();
        this.tokenTrie = new SearchTrie();
        this.nextId=1;
        tokenSet.add(END_OF_INPUT = new Token(0, null));//Add end of input token
    }

    public List<Token> tokenize(String input) {
        List<Token>tokenList = new LinkedList<>();
        StringBuilder extract = new StringBuilder(100);
        for (int i = 0; i < input.length(); i++) {
            extract.append(input.charAt(i));
            SearchTrie.QueryResponse response = tokenTrie.query(extract.toString());
            if(extract.length()==1 && response.hasPath()==false) {
                //This means we cant tokenize this string so throw an exception
                throw new RuntimeException("Unable to tokenize: " + extract.toString());
            }
            if(response.hasPath()) {
                if(i==(input.length()-1)) {
                    //If valid token, then emit
                    if(response.getId()!=-1) {
                        tokenList.add(tokenIndexById.get(response.getId()));
                        return tokenList;
                    } else {
                        i = backStep(extract, tokenList, i);
                    }
                } else {
                    continue;
                }
            } else {
                i = backStep(extract, tokenList, i);
            }
        }
        return tokenList;
    }

    public int backStep(StringBuilder extract, List<Token>tokenList, int i) {
        int numBackstep = 0;
        long id = -1;
        while(id==-1) {
            extract.setLength(extract.length()-1);
            numBackstep++;
            id = tokenTrie.query(extract.toString()).getId();
        }
        extract.setLength(0);
        tokenList.add(tokenIndexById.get(id));
        return i - numBackstep;
    }

    public void initializeFromText(String text) {
        for (int i = 0; i < text.length(); i++) {
            createNewToken(Character.toString(text.charAt(i)));
        }
    }

    public void createNewToken(String tokenString) {
        if(null == tokenString) {
            return;
        }
        SearchTrie.QueryResponse response = tokenTrie.query(tokenString);
        if(response.getId()==-1) {
            Token token = new Token(nextId++, tokenString);
            tokenSet.add(token);
            tokenIndexById.put(token.getId(), token);
            tokenTrie.add(token.getToken(), token.getId());
        }
    }

    public void loadFromFile() {
        File tokenFile = new File(fileName);
        if(!tokenFile.exists()) {
            return;
        }
        int length = (int)(tokenFile.length());
        byte[]data = new byte[length];
        try(InputStream is = new BufferedInputStream(new FileInputStream(tokenFile), 8096)) {
            is.read(data);
        } catch(Exception e) {
            throw new RuntimeException("Unable to load file", e);
        }
        tokenSet.clear();
        tokenIndexById.clear();
        tokenTrie.clear();
        Token[]tokenArray = gson.fromJson(new String(data, StandardCharsets.UTF_8), Token[].class);
        long max = -1;
        for (Token token : tokenArray) {
            if(token.getId() == 0) {
                END_OF_INPUT = token;
            }
            tokenSet.add(token);
            tokenIndexById.put(token.getId(), token);
            tokenTrie.add(token.getToken(), token.getId());
            if(max<token.getId()) {
                max = token.getId();
            }
        }
        nextId = max + 1;
    }

    public void saveToFile() {
        List<Token>tokenList = new LinkedList<>();
        tokenList.addAll(tokenSet);
        tokenList.sort((o1, o2)->Long.compare(o1.getId(), o2.getId()));
        String data = gson.toJson(tokenList);
        try (OutputStream os = new BufferedOutputStream(new FileOutputStream(fileName), 8096)) {
            os.write(data.getBytes(StandardCharsets.UTF_8));
        } catch(Exception e) {
            throw new RuntimeException("Unable to load file", e);
        }
    }

    public int size() {
        return tokenSet.size();
    }
}
