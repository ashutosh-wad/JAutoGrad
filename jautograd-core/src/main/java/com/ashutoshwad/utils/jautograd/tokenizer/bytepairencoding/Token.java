package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import java.io.*;

public class Token implements Serializable {
    private long id;
    private String token;

    public Token() {}

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

    /*
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeLong(id);
        boolean isTokenNull = null == token;
        out.writeBoolean(isTokenNull);
        if(!isTokenNull) {
            out.writeUTF(token);
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        id = in.readLong();
        boolean isTokenNull = in.readBoolean();
        if(isTokenNull) {
            token = null;
        } else {
            token = in.readUTF();
        }
    }
    */
}
