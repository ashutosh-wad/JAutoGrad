package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import java.util.Objects;

public class TokenPair implements Comparable<TokenPair> {
    private final Token one;
    private final Token two;

    public TokenPair(Token one, Token two) {
        this.one = Objects.requireNonNull(one);
        this.two = Objects.requireNonNull(two);
    }

    public Token getOne() {
        return one;
    }

    public Token getTwo() {
        return two;
    }

    @Override
    public boolean equals(Object o) {
        /*
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TokenPair tokenPair = (TokenPair) o;
        return Objects.equals(one.getId(), tokenPair.one.getId()) && Objects.equals(two.getId(), tokenPair.two.getId());
        */
        TokenPair tokenPair = (TokenPair) o;
        return one.getId()==tokenPair.one.getId() && two.getId() == tokenPair.two.getId();
    }

    @Override
    public int hashCode() {
        return (int)(31*one.getId()*two.getId());
        //return Objects.hash(, two.getId());
    }

    @Override
    public String toString() {
        return one.toString() + "|" + two.toString();
    }

    @Override
    public int compareTo(TokenPair o) {
        if(null==o) {
            //Anything is greater than null
            return 1;
        }
        int retVal = Long.compare(one.getId(), o.one.getId());
        if(0==retVal) {
            retVal = Long.compare(two.getId(), o.two.getId());
        }
        return retVal;
    }
}
