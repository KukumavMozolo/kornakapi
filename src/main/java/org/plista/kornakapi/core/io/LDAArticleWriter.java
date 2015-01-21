package org.plista.kornakapi.core.io;

import org.plista.kornakapi.core.config.Configuration;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.web.Components;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class LDAArticleWriter {
	
	  private static final Logger log = LoggerFactory.getLogger(LDAArticleWriter.class);
	public void writeArticle(String pLabel, long pItemId, String pText) throws IOException {
		Components components = Components.instance();
		Configuration config = components.getConfiguration();
    	String documentsPath = (((LDARecommenderConfig) config.getLDARecommender()).getTextDirectoryPath()+  Long.toString(pItemId));

    	write(documentsPath, pText);
    	
	}
	private void write(String filename, String pText) throws IOException{
    	File f = new File(filename);
    	if(f.exists()){
    		f.delete();
    	}
		f.createNewFile();
		BufferedWriter output = new BufferedWriter(new FileWriter(f));
        output.write(pText);
        output.close();
	}
}
