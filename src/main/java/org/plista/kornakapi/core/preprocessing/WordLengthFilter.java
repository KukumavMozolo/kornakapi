package org.plista.kornakapi.core.preprocessing;

import java.io.IOException;
import java.io.StringWriter;
import java.util.StringTokenizer;

/**
 * Created by mx on 4/15/15.
 */
public class WordLengthFilter{

    /**
     * Removes all words <= minWordLength from string
     *
     * @param pInput
     * @return
     * @throws IOException
     */
    public String filterText(String pInput, int minWordLength) throws IOException {
        pInput = pInput.toLowerCase();
        StringWriter sw = new StringWriter();
        StringTokenizer st = new StringTokenizer(pInput);
        while(st.hasMoreTokens()) {
            String token = st.nextToken();
            if(token.length() >= minWordLength) {
                sw.write(token);
                sw.write(' ');
            }
        }
        return sw.toString().trim();
    }
}
