package com.ashutoshwad.utils.jautograd.tokenizer.bytepairencoding;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class SearchTrieTest {
    @Test
    public void testTrie() {
        SearchTrie trie = new SearchTrie();
        trie.add("a", 1L);
        trie.add("b", 2L);
        trie.add("c", 3L);
        trie.add("abc", 4L);

        assertTrue(trie.query("a").hasPath());
        assertTrue(trie.query("b").hasPath());
        assertTrue(trie.query("c").hasPath());
        assertTrue(trie.query("ab").hasPath());
        assertTrue(trie.query("abc").hasPath());
        assertFalse(trie.query("d").hasPath());
        assertFalse(trie.query("da").hasPath());
        assertFalse(trie.query("ad").hasPath());

        assertEquals(1, trie.query("a").getId());
        assertEquals(2, trie.query("b").getId());
        assertEquals(3, trie.query("c").getId());
        assertEquals(-1, trie.query("ab").getId());
        assertEquals(4, trie.query("abc").getId());
        assertEquals(-1, trie.query("d").getId());
        assertEquals(-1, trie.query("da").getId());
        assertEquals(-1, trie.query("ad").getId());
    }
}
