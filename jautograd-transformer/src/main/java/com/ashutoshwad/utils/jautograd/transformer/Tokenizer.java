package com.ashutoshwad.utils.jautograd.transformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * There are a total of 88 supported tokens in this tokenizer. Unknown tokens are replaced with spaces.
 */
public class Tokenizer {
    private Map<String, List<Integer>> tokCache = new HashMap<>();

    public int mapFromValidToken(char token) {
        switch (token) {
            case 10:
                return 0;
            case 32:
                return 1;
            case 33:
                return 2;
            case 34:
                return 3;
            case 36:
                return 4;
            case 37:
                return 5;
            case 38:
                return 6;
            case 39:
                return 7;
            case 40:
                return 8;
            case 41:
                return 9;
            case 42:
                return 10;
            case 44:
                return 11;
            case 45:
                return 12;
            case 46:
                return 13;
            case 47:
                return 14;
            case 48:
                return 15;
            case 49:
                return 16;
            case 50:
                return 17;
            case 51:
                return 18;
            case 52:
                return 19;
            case 53:
                return 20;
            case 54:
                return 21;
            case 55:
                return 22;
            case 56:
                return 23;
            case 57:
                return 24;
            case 58:
                return 25;
            case 59:
                return 26;
            case 61:
                return 27;
            case 63:
                return 28;
            case 65:
                return 29;
            case 66:
                return 30;
            case 67:
                return 31;
            case 68:
                return 32;
            case 69:
                return 33;
            case 70:
                return 34;
            case 71:
                return 35;
            case 72:
                return 36;
            case 73:
                return 37;
            case 74:
                return 38;
            case 75:
                return 39;
            case 76:
                return 40;
            case 77:
                return 41;
            case 78:
                return 42;
            case 79:
                return 43;
            case 80:
                return 44;
            case 81:
                return 45;
            case 82:
                return 46;
            case 83:
                return 47;
            case 84:
                return 48;
            case 85:
                return 49;
            case 86:
                return 50;
            case 87:
                return 51;
            case 88:
                return 52;
            case 89:
                return 53;
            case 90:
                return 54;
            case 91:
                return 55;
            case 92:
                return 56;
            case 93:
                return 57;
            case 94:
                return 58;
            case 95:
                return 59;
            case 97:
                return 60;
            case 98:
                return 61;
            case 99:
                return 62;
            case 100:
                return 63;
            case 101:
                return 64;
            case 102:
                return 65;
            case 103:
                return 66;
            case 104:
                return 67;
            case 105:
                return 68;
            case 106:
                return 69;
            case 107:
                return 70;
            case 108:
                return 71;
            case 109:
                return 72;
            case 110:
                return 73;
            case 111:
                return 74;
            case 112:
                return 75;
            case 113:
                return 76;
            case 114:
                return 77;
            case 115:
                return 78;
            case 116:
                return 79;
            case 117:
                return 80;
            case 118:
                return 81;
            case 119:
                return 82;
            case 120:
                return 83;
            case 121:
                return 84;
            case 122:
                return 85;
            case 124:
                return 86;
            case 125:
                return 87;
            case 126:
                return 88;
            default:
                return 32;
        }
    }

    public char mapToValidToken(int token) {
        switch (token) {
            case 0: return 10;
            case 1: return 32;
            case 2: return 33;
            case 3: return 34;
            case 4: return 36;
            case 5: return 37;
            case 6: return 38;
            case 7: return 39;
            case 8: return 40;
            case 9: return 41;
            case 10: return 42;
            case 11: return 44;
            case 12: return 45;
            case 13: return 46;
            case 14: return 47;
            case 15: return 48;
            case 16: return 49;
            case 17: return 50;
            case 18: return 51;
            case 19: return 52;
            case 20: return 53;
            case 21: return 54;
            case 22: return 55;
            case 23: return 56;
            case 24: return 57;
            case 25: return 58;
            case 26: return 59;
            case 27: return 61;
            case 28: return 63;
            case 29: return 65;
            case 30: return 66;
            case 31: return 67;
            case 32: return 68;
            case 33: return 69;
            case 34: return 70;
            case 35: return 71;
            case 36: return 72;
            case 37: return 73;
            case 38: return 74;
            case 39: return 75;
            case 40: return 76;
            case 41: return 77;
            case 42: return 78;
            case 43: return 79;
            case 44: return 80;
            case 45: return 81;
            case 46: return 82;
            case 47: return 83;
            case 48: return 84;
            case 49: return 85;
            case 50: return 86;
            case 51: return 87;
            case 52: return 88;
            case 53: return 89;
            case 54: return 90;
            case 55: return 91;
            case 56: return 92;
            case 57: return 93;
            case 58: return 94;
            case 59: return 95;
            case 60: return 97;
            case 61: return 98;
            case 62: return 99;
            case 63: return 100;
            case 64: return 101;
            case 65: return 102;
            case 66: return 103;
            case 67: return 104;
            case 68: return 105;
            case 69: return 106;
            case 70: return 107;
            case 71: return 108;
            case 72: return 109;
            case 73: return 110;
            case 74: return 111;
            case 75: return 112;
            case 76: return 113;
            case 77: return 114;
            case 78: return 115;
            case 79: return 116;
            case 80: return 117;
            case 81: return 118;
            case 82: return 119;
            case 83: return 120;
            case 84: return 121;
            case 85: return 122;
            case 86: return 124;
            case 87: return 125;
            case 88: return 126;
            default:
                return ' ';
        }
    }

    public void convertFromSourceFile(char ch, List<Integer>tokenList) {
        if(ch<128) {
            tokenList.add(mapFromValidToken(ch));
            return;
        }
        switch (ch) {
            case 169:
                addString(tokenList, " (Copyright) ");
                break;
            case 171:
                addString(tokenList, " << ");
                break;
            case 174:
                addString(tokenList, " (Registered Trademark)");
                break;
            case 176:
                addString(tokenList, " Degrees ");
                break;
            case 187:
                addString(tokenList, " >> ");
                break;
            case 215:
                addString(tokenList, " x ");
                break;
            case 231:
                addString(tokenList, " c ");
                break;
            case 233:
                addString(tokenList, " e ");
                break;
            case 239:
                addString(tokenList, " i ");
                break;
            case 321:
                addString(tokenList, " l ");
                break;
            case 8211:
                addString(tokenList, "-");
                break;
            case 8212:
                addString(tokenList, "--");
                break;
            case 8216:
                addString(tokenList, "'");
                break;
            case 8217:
                addString(tokenList, "'");
                break;
            case 8220:
                addString(tokenList, "\"");
                break;
            case 8221:
                addString(tokenList, "\"");
                break;
            case 8226:
                addString(tokenList, ".");
                break;
            case 8230:
                addString(tokenList, "...");
                break;
            case 8482:
                addString(tokenList, " (Trademark) ");
                break;
            default:
                tokenList.add(32);
        }
    }

    private void addString(List<Integer> retlist, String val) {
        if (!tokCache.containsKey(val)) {
            List<Integer> cachedList = new ArrayList<>();
            char[] chars = val.toCharArray();
            for (char ch : chars) {
                cachedList.add(mapFromValidToken(ch));
            }
            tokCache.put(val, cachedList);
        }
        retlist.addAll(tokCache.get(val));
    }
}
