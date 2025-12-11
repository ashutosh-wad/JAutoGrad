package com.ashutoshwad.utils.jautograd;

public class StringUtils {
    public static final boolean isBlank(String str) {
        if (null == str) {
            return true;
        }
        return str.trim().isEmpty();
    }
}
