package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

public class Token {
    private long id;
    private String token;

    public Token(long id, String token) {
        this.id = id;
        this.token = token;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getToken() {
        return token;
    }

    public void setToken(String token) {
        this.token = token;
    }

    @Override
    public String toString() {
        if(null==token) {
            return "<null>";
        }
        return token.replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\"", "\\\"");
    }
}
