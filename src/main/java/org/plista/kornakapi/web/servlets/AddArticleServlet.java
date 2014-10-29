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

package org.plista.kornakapi.web.servlets;
import org.apache.hadoop.fs.Path;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.io.LDAArticleWriter;
import org.plista.kornakapi.core.preprocessing.BadcharFilter;
import org.plista.kornakapi.core.preprocessing.StopwordFilter;
import org.plista.kornakapi.core.training.DocumentTopicInferenceTrainer;
import org.plista.kornakapi.web.Parameters;
import org.quartz.SchedulerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/** servlet to add articles to a candidate set */
public class AddArticleServlet extends BaseServlet {
	
	private static final Logger log = LoggerFactory.getLogger(AddArticleServlet.class);

  @Override
  protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
	  
	String label = getParameter(request, Parameters.LABEL, true);
    String text = getParameter(request, Parameters.Text, true);
    long itemID = getParameterAsLong(request, Parameters.ITEM_ID, true);
    if(itemID < 0 || itemID > 2147483647){
    	itemID = this.idRemapping(itemID);
    }
    try{
 
    	this.storages().get("lda").addCandidate(label, itemID);
    	LDAArticleWriter writer = new LDAArticleWriter();
    	
    	// save the fulltext as usual
    	writer.writeArticle(label, itemID, text, "pure");
    	
    	// save preprocessed text 
    	StopwordFilter filter = new StopwordFilter();
    	BadcharFilter filter_bc = new BadcharFilter();
    	
    	String processed = filter.filterText(filter_bc.filterText(text));
    	writer.writeArticle(label, itemID, processed, "stopwords");
    	
    	topicInferenceForNewItems();	
    } catch(NullPointerException e){
	  if(log.isInfoEnabled()){
		  log.info("No Recommender found for label {} and itemID {}", label, itemID );
	  }
    }
  }
  /**
   * 
   * @param name
   * @param itemid
   */
  private void topicInferenceForItem(String label, String itemid){
	  String name = itemid+ "_lda";
	  LDARecommenderConfig conf = (LDARecommenderConfig) this.getConfiguration().getLDARecommender();
	  Path p = new Path(conf.getLDARecommenderModelPath());
	  DocumentTopicInferenceTrainer trainer = new DocumentTopicInferenceTrainer(conf, p);
	  this.setTrainer(name, trainer);
      scheduler().addRecommenderTrainingJob(name);
      try {
		scheduler().immediatelyTrainRecommender(name);
//		this.storages().get("0").addCandidate(label, Long.parseLong(itemid));
	} catch (SchedulerException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	} catch (NumberFormatException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }
  
  /**
   * This methods tries to create a new job that performs topicInference for new articles
   * All articles in the respective folder will be considered.
   */
  private void topicInferenceForNewItems(){
	  String name = "inference_lda";
	  LDARecommenderConfig conf = (LDARecommenderConfig) this.getConfiguration().getLDARecommender();
	  Path p = new Path(conf.getLDARecommenderModelPath());
	  DocumentTopicInferenceTrainer trainer = new DocumentTopicInferenceTrainer(conf, p);
	  this.setTrainer(name, trainer);
      scheduler().addRecommenderTrainingJob(name);
      try {
		scheduler().immediatelyTrainRecommender(name);
	} catch (SchedulerException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }
}
