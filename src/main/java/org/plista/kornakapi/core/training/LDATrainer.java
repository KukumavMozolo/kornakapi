/**
 * Copyright 2012 plista GmbH  (http://www.plista.com/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and limitations under the License.
 */


package org.plista.kornakapi.core.training;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;

import com.google.common.collect.Lists;
import com.google.common.io.Files;

public class LDATrainer extends AbstractTrainer{
	
	private LDARecommenderConfig conf;
	
	public LDATrainer(RecommenderConfig conf){
		super(conf);
		this.conf = (LDARecommenderConfig) conf;
	}

	@Override
	protected void doTrain(File targetFile, DataModel inmemoryData,
			int numProcessors) throws IOException {
		try {
			collectNewArticles();
			new FromDirectoryVectorizer(conf).doTrain();
			new LDATopicModeller(conf).doTrain();
			printLocalTopicWordDistribution(conf,((LDARecommenderConfig)conf).getTopicsOutputPath(),((LDARecommenderConfig)conf).getTopicsOutputPath());
			printLocalDocumentTopicDistribution(conf,((LDARecommenderConfig)conf).getLDADocTopicsPath(),((LDARecommenderConfig)conf).getLDADocTopicsPath());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	/**
	 * copys all new articles to the corpus
	 */
	protected void collectNewArticles(){
		// train a specific training set
		String trainingSet = conf.getTrainingSetName();
		File newDocs = new File(((LDARecommenderConfig)conf).getInferencePath()+ "Documents/" + trainingSet + '/' );
		String corpusDir = ((LDARecommenderConfig)conf).getTextDirectoryPath();
		for(File from: newDocs.listFiles()){
			File to = new File(corpusDir+ from.getName());
			try {
				File newFile = new File(to.toString());
				if(newFile.exists()){
					newFile.delete();
				}
				Files.move(from, to);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}	
	}
	
	/**
	 * Dumps the Topic distributen to file
	 * @param conf
	 * @param input
	 * @param output
	 */
	public static void printLocalTopicWordDistribution(RecommenderConfig conf, String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input);
	        argList.add("-o");
	        argList.add("/opt/kornakapi-model/lda/print/topics.txt");
	        argList.add("--dictionaryType");
	        argList.add("sequencefile");
	        argList.add("-d");
	        argList.add(((LDARecommenderConfig)conf).getTopicsDictionaryPath());
	        argList.add("-sort");
	        argList.add("true");
	        argList.add("-vs");
	        argList.add("100");
	        String[] args = argList.toArray(new String[argList.size()]);
	        try {
				//LDAPrintTopics.main(args);
				VectorDumper.main(args);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	
	/**
	 * 
	 * @param conf
	 * @param input
	 * @param output
	 */
	public static void printLocalDocumentTopicDistribution(RecommenderConfig conf, String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input);
	        argList.add("-o");
	        argList.add("/opt/kornakapi-model/lda/print/DocumentTopics.txt");
	        argList.add("-sort");
	        argList.add("true");
	        argList.add("-vs");
	        argList.add("100");
	        argList.add("-p");
	        argList.add("true");
	        String[] args = argList.toArray(new String[argList.size()]);
	        try {
				//LDAPrintTopics.main(args);
				VectorDumper.main(args);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}	
}
