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

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.dbcp.BasicDataSource;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.config.*;
import org.plista.kornakapi.core.recommender.CachingAllUnknownItemsCandidateItemsStrategy;
import org.plista.kornakapi.core.recommender.FoldingFactorizationBasedRecommender;
import org.plista.kornakapi.core.recommender.ItemSimilarityBasedRecommender;
import org.plista.kornakapi.core.recommender.LDATopicRecommender;
import org.plista.kornakapi.core.recommender.factory.FFBRFactory;
import org.plista.kornakapi.core.recommender.factory.ISBRFactory;
import org.plista.kornakapi.core.storage.CandidateCacheStorageDecorator;
import org.plista.kornakapi.core.storage.MySqlMaxPersistentStorage;
import org.plista.kornakapi.core.storage.MySqlStorage;
import org.plista.kornakapi.core.storage.SemanticMySqlStorage;
import org.plista.kornakapi.core.training.factory.LDATrainerFactory;
import org.plista.kornakapi.core.training.*;
import org.plista.kornakapi.core.training.preferencechanges.DelegatingPreferenceChangeListenerForLabel;
import org.plista.kornakapi.core.training.preferencechanges.InMemoryPreferenceChangeListener;
import org.plista.kornakapi.web.Components;
import org.quartz.SchedulerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/** servlet context listener to initialize/shut down the application */
public class BigBangServletContextListener implements ServletContextListener {

  private static final String CONFIG_PROPERTY = "kornakapi.conf";

  private static final Logger log = LoggerFactory.getLogger(BigBangServletContextListener.class);
  
  Map<String, KornakapiRecommender> recommenders;
  Map<String, Trainer> trainers;
  TaskScheduler scheduler;
  DelegatingPreferenceChangeListenerForLabel preferenceChangeListener;
  HashMap<String, DataModel> persitentDatas;
  HashMap<String, CandidateCacheStorageDecorator> storages;
  LinkedList<String> labels;
  Configuration conf;
  BasicDataSource dataSource;
  CandidateCacheStorageDecorator domainIndependetStorage;
  


  @Override
  public void contextInitialized(ServletContextEvent event) {
	  try{
	      log.info("Try started");
	      String configFileLocation = System.getProperty(CONFIG_PROPERTY);
	      Preconditions.checkState(configFileLocation != null, "configuration file not set!");
	
	      File configFile = new File(configFileLocation);
	      Preconditions.checkState(configFile.exists() && configFile.canRead(),
	          "configuration file not found or not readable");
	
	      conf = Configuration.fromXML(Files.toString(configFile, Charsets.UTF_8));
	
	      Preconditions.checkState(conf.getNumProcessorsForTraining() > 0, "need at least one processor for training!");
	      domainIndependetStorage = null;
	      labels = null;
	
	      dataSource = new BasicDataSource();
	      storages = new HashMap<String, CandidateCacheStorageDecorator>();
	      if(conf.getMaxPersistence()){
	    	  domainIndependetStorage = new CandidateCacheStorageDecorator( new MySqlMaxPersistentStorage(conf.getStorageConfiguration(), "",dataSource));
	          labels = domainIndependetStorage.getAllLabels();
	    	  for(String label: labels){
	    		  storages.put(label, new CandidateCacheStorageDecorator(new MySqlMaxPersistentStorage(conf.getStorageConfiguration(), label,dataSource)));
	    	  }
	      }else{
	    	  domainIndependetStorage = new CandidateCacheStorageDecorator( new MySqlStorage(conf.getStorageConfiguration(), "",dataSource));
	          labels = domainIndependetStorage.getAllLabels();
	    	  for(String label: labels){
	    		  storages.put(label,  new CandidateCacheStorageDecorator(new MySqlStorage(conf.getStorageConfiguration(), label,dataSource)));
	    	  }
	      }   
		  persitentDatas = new HashMap<String, DataModel>();
		  for(String label: labels){
		      persitentDatas.put(label, storages.get(label).recommenderData());
		  }
	
	
	      scheduler = new TaskScheduler();
	
	      String purgePreferencesCronExpression = conf.getStorageConfiguration().getPurgePreferencesCronExpression();
	
	      scheduler.setPurgeOldPreferences(purgePreferencesCronExpression);
	
	      recommenders = Maps.newHashMap();
	      trainers = Maps.newHashMap();
	
	      preferenceChangeListener = new DelegatingPreferenceChangeListenerForLabel();
	      
	      setupRecommenders();
      
      } catch (Exception e) {
     	log.info("Something bad happend: {}" , e.getMessage());
     }
  }
  
  
    private void setupRecommenders() throws SchedulerException, IOException, TasteException{
      log.info("Setup itemBasedRecommders");
      ISBRFactory isbrFactory = new ISBRFactory();
      for (ItembasedRecommenderConfig itembasedConf : conf.getItembasedRecommenders()) {
    	for(String label: labels){
    	       String name = itembasedConf.getName() +"_"+ label;
    	       	ItemSimilarityBasedRecommender recommender = isbrFactory.getRecommender(conf, persitentDatas.get(label), name);
    	        recommenders.put(name, recommender);
    	        putRecommender(recommender, name);
    	        putTrainer(new MultithreadedItembasedInMemoryTrainer(itembasedConf), itembasedConf, name, label);
    	        trainers.put(name, new MultithreadedItembasedInMemoryTrainer(itembasedConf));
    	        log.info("Created ItemBasedRecommender [{}] using similarity [{}] and [{}] similar items per item",
    	            new Object[] { name, itembasedConf.getSimilarityClass(), itembasedConf.getSimilarItemsPerItem() });
    	}
 
      }     
      
      log.info("Setup FactorizationBasedRecommders");
      FFBRFactory ffbrFactory = new FFBRFactory();
      for (FactorizationbasedRecommenderConfig factorizationbasedConf : conf.getFactorizationbasedRecommenders()) {
    	  for(String label: labels){
    	  	String name = factorizationbasedConf.getName() +"_"+ label;
    	    FoldingFactorizationBasedRecommender svdRecommender = ffbrFactory.getRecommender(conf, factorizationbasedConf, persitentDatas.get(label), label, name);
    	    putRecommender(svdRecommender,  name);
    	    putTrainer(new FactorizationbasedInMemoryTrainer(factorizationbasedConf), factorizationbasedConf, name, label);
	        log.info("Created FactorizationBasedRecommender [{}] using [{}] features and [{}] iterations",
	            new Object[] { name, factorizationbasedConf.getNumberOfFeatures(),
	                factorizationbasedConf.getNumberOfIterations() });
    	  }
  
      }
      log.info("Setup LDARecommender");
      String name = "lda";
      LDARecommenderConfig ldaconf = (LDARecommenderConfig) conf.getLDARecommender();
	  BasicDataSource dataSource = new BasicDataSource();
	  CandidateCacheStorageDecorator dec =new CandidateCacheStorageDecorator(new SemanticMySqlStorage(conf.getStorageConfiguration(), name,dataSource)); 
	  DataModel dmodel = dec.recommenderData();
	  CandidateItemsStrategy allUnknownItemsStrategy =
		           new CachingAllUnknownItemsCandidateItemsStrategy(dmodel);
	  LDATopicRecommender recommender = new LDATopicRecommender(dmodel, allUnknownItemsStrategy , ldaconf);
	  putRecommender(recommender,  name);
      LDATrainerFactory ldaFactory = new LDATrainerFactory(ldaconf);
	  putTrainer(ldaFactory.getTrainer(), conf.getLDARecommender(), name, "doesNotMatter");
      log.info("Created LDARecommender");
      storages.put(name,  dec);
	  
	  
      
      /**
      log.info("Setup KluserRecommders");
      for (StreamingKMeansClustererConfig streamingKMeansClustererConf : conf.getStreamingKMeansClusterer()) {
      	for(String label: labels){
         	  String name = streamingKMeansClustererConf.getName() +"_"+ label;
        	  
              File modelFile = new File(conf.getModelDirectory(), name + ".model");

              PersistenceStrategy persistence = new FilePersistenceStrategy(modelFile);

              if (!modelFile.exists()) {
                createEmptyFactorization(persistence);
              }
              StreamingKMeansClassifierModel model = new StreamingKMeansClassifierModel(conf.getStorageConfiguration(),label,dataSource); 
              StreamingKMeansClustererTrainer clusterer = new StreamingKMeansClustererTrainer( streamingKMeansClustererConf, model);
              trainers.put(name,clusterer);
              
              StreamingKMeansClassifierRecommender recommender = new StreamingKMeansClassifierRecommender(model);
              recommenders.put(name, recommender);
              
              String cronExpression = streamingKMeansClustererConf.getRetrainCronExpression();
              if (cronExpression == null) {
                scheduler.addRecommenderTrainingJob(name);
              } else {
                scheduler.addRecommenderTrainingJobWithCronSchedule(name, cronExpression);
              }
              
              if (streamingKMeansClustererConf.getRetrainAfterPreferenceChanges() !=
                      RecommenderConfig.DONT_RETRAIN_ON_PREFERENCE_CHANGES) {
                    preferenceChangeListener.addDelegate(new InMemoryPreferenceChangeListener(scheduler, name,
                    		streamingKMeansClustererConf.getRetrainAfterPreferenceChanges()));
                  }
              
              log.info("Created StreamingKMeansClusterer [{}] with [{}] minclusters and [{}] cutoff distance",
                  new Object[] { name, streamingKMeansClustererConf.getDesiredNumCluster(), streamingKMeansClustererConf.getDistanceCutoff()}); 
      	}
 
      }
      **/
      
      
      log.info("Initialize Components");
      Components.init(conf, storages, recommenders, trainers, scheduler, preferenceChangeListener, labels, dataSource, domainIndependetStorage);
      log.info("Start Scheduler");
      scheduler.start();

    } 
  




  @Override
  public void contextDestroyed(ServletContextEvent event) {
    Components components = Components.instance();
    LinkedList<String> labels = components.getLabels();
    for(String label: labels){
        Closeables.closeQuietly(components.storages().get(label));
    }
    Closeables.closeQuietly(components.scheduler());
    Closeables.closeQuietly(components.getDomainIndependetStorage());
  }
  private void putRecommender(KornakapiRecommender recommender, String recommenderName) {
      recommenders.put(recommenderName, recommender);
  }
  private void putTrainer(AbstractTrainer trainer,  RecommenderConfig factorizationbasedConf, String recommenderName,String label ) throws SchedulerException{
      trainers.put(recommenderName, trainer);
      String cronExpression = factorizationbasedConf.getRetrainCronExpression();
      if (cronExpression == null) {
        scheduler.addRecommenderTrainingJob(recommenderName);
      } else {
        scheduler.addRecommenderTrainingJobWithCronSchedule(recommenderName, cronExpression);
        scheduler.immediatelyTrainRecommender(recommenderName);
      }

      if (factorizationbasedConf.getRetrainAfterPreferenceChanges() !=
          RecommenderConfig.DONT_RETRAIN_ON_PREFERENCE_CHANGES) {
        preferenceChangeListener.addDelegate(new InMemoryPreferenceChangeListener(scheduler, recommenderName,
            factorizationbasedConf.getRetrainAfterPreferenceChanges()), label);
      }
  }
}
