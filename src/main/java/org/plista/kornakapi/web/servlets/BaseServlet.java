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

import org.apache.commons.dbcp.BasicDataSource;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.config.Configuration;
import org.plista.kornakapi.core.config.FactorizationbasedRecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.plista.kornakapi.core.recommender.CachingAllUnknownItemsCandidateItemsStrategy;
import org.plista.kornakapi.core.recommender.FoldingFactorizationBasedRecommender;
import org.plista.kornakapi.core.storage.CandidateCacheStorageDecorator;
import org.plista.kornakapi.core.storage.MySqlMaxPersistentStorage;
import org.plista.kornakapi.core.storage.MySqlStorage;
import org.plista.kornakapi.core.training.FactorizationbasedInMemoryTrainer;
import org.plista.kornakapi.core.training.TaskScheduler;
import org.plista.kornakapi.core.training.Trainer;
import org.plista.kornakapi.core.training.preferencechanges.DelegatingPreferenceChangeListenerForLabel;
import org.plista.kornakapi.core.training.preferencechanges.InMemoryPreferenceChangeListener;
import org.plista.kornakapi.web.Components;
import org.plista.kornakapi.web.InvalidParameterException;
import org.plista.kornakapi.web.MissingParameterException;
import org.quartz.SchedulerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;

/** base class for all servlets */
public abstract class BaseServlet extends HttpServlet {

  private static final Pattern ITEM_ID_SEPARATOR = Pattern.compile(",");
	private static final Logger log = LoggerFactory.getLogger(TrainServlet.class);

  private Components getComponents() {
    return Components.instance();
  }
  
  protected Configuration getConfiguration(){
	  return getComponents().getConfiguration();
  }
  
  protected void setRecommender(String name, KornakapiRecommender recommender){
	  getComponents().setRecommender(name, recommender);
  }
  
  protected BasicDataSource getDataSource(){
	  return getComponents().getDataSource();
  }

  protected KornakapiRecommender recommender(String name) {
    return getComponents().recommender(name);
  }

  protected void setTrainer(String name, Trainer trainer){
	 getComponents().setTrainer(name, trainer);
  }
  protected boolean containsTrainer(String name){
	  return getComponents().trainer(name) != null;
  }

  protected TaskScheduler scheduler() {
    return getComponents().scheduler();
  }

  protected HashMap<String, CandidateCacheStorageDecorator> storages() {
    return getComponents().storages();
  }

  protected DelegatingPreferenceChangeListenerForLabel preferenceChangeListener() {
    return getComponents().preferenceChangeListener();
  }

  protected boolean hasParameter(HttpServletRequest request, String name) {
    return request.getParameter(name) != null;
  }

  protected String getParameter(HttpServletRequest request, String name, boolean required) {
    String param = request.getParameter(name);

    if (param == null && required) {
      throw new MissingParameterException("Parameter [" + name + "] must be supplied!");
    }

    return param;
  }

  protected long getParameterAsLong(HttpServletRequest request, String name, boolean required) {
    String param = getParameter(request, name, required);

    try {
      return Long.parseLong(param);
    } catch (NumberFormatException e) {
      throw new InvalidParameterException("Unable to parse parameter [" + name + "]", e);
    }
  }
  
  protected CandidateCacheStorageDecorator getDomainIndependetStorage(){
	  return getComponents().getDomainIndependetStorage();
  }

  protected long[] getParameterAsLongArray(HttpServletRequest request, String name) {
    String param = getParameter(request, name, false);

    String[] tokens = ITEM_ID_SEPARATOR.split(param);
    long[] itemIDs = new long[tokens.length];

    for (int n = 0; n < itemIDs.length; n++) {
      try {
        itemIDs[n] = Long.parseLong(tokens[n]);
      } catch (NumberFormatException e) {
        throw new InvalidParameterException("Unable to parse parameter [" + name + "]", e);
      }
    }

    return itemIDs;
  }

  protected float getParameterAsFloat(HttpServletRequest request, String name, boolean required) {
    String param = getParameter(request, name, required);

    try {
      return Float.parseFloat(param);
    } catch (NumberFormatException e) {
      throw new InvalidParameterException("Unable to parse parameter [" + name + "]", e);
    }
  }

  protected int getParameterAsInt(HttpServletRequest request, String name, int defaultValue) {
    String param = getParameter(request, name, false);

    try {
      return param != null ? Integer.parseInt(param) : defaultValue;
    } catch (NumberFormatException e) {
      throw new InvalidParameterException("Unable to parse parameter [" + name + "]", e);
    }
  }
  protected void createRecommenderForLabel(String label) throws IOException, TasteException{
	  	Configuration conf = getConfiguration();
	  	List<FactorizationbasedRecommenderConfig> factorizationbasedConfs= conf.getFactorizationbasedRecommenders();
      FactorizationbasedRecommenderConfig factorizationbasedConf =factorizationbasedConfs.get(0);

      String name = factorizationbasedConf.getName() +"_"+ label;
      if(conf.getMaxPersistence()){
      	storages().put(label, new CandidateCacheStorageDecorator(new MySqlMaxPersistentStorage(conf.getStorageConfiguration(), label,getDataSource())));
      }else{
    		storages().put(label,  new CandidateCacheStorageDecorator(new MySqlStorage(conf.getStorageConfiguration(), label,getDataSource())));

      }

      File modelFile = new File(conf.getModelDirectory(), name + ".model");

      PersistenceStrategy persistence = new FilePersistenceStrategy(modelFile);

      if (!modelFile.exists()) {
        createEmptyFactorization(persistence);
      }

      DataModel persistenData = storages().get(label).recommenderData();
      
      CandidateItemsStrategy allUnknownItemsStrategy =
          new CachingAllUnknownItemsCandidateItemsStrategy(persistenData);

      FoldingFactorizationBasedRecommender svdRecommender = new FoldingFactorizationBasedRecommender(persistenData,
          allUnknownItemsStrategy, persistence, conf.getFactorizationbasedRecommenders().get(0).getNumberOfThreadsForEstimation());

      setRecommender(name, svdRecommender);
      setTrainer(name, new FactorizationbasedInMemoryTrainer(factorizationbasedConf));

      String cronExpression = factorizationbasedConf.getRetrainCronExpression();
      if (cronExpression == null) {
      	scheduler().addRecommenderTrainingJob(name);
      } else {
      	scheduler().addRecommenderTrainingJobWithCronSchedule(name, cronExpression);
      	try {
				scheduler().immediatelyTrainRecommender(name);
			} catch (SchedulerException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
      }

      if (factorizationbasedConf.getRetrainAfterPreferenceChanges() !=
          RecommenderConfig.DONT_RETRAIN_ON_PREFERENCE_CHANGES) {
      	((DelegatingPreferenceChangeListenerForLabel)preferenceChangeListener()).addDelegate(new InMemoryPreferenceChangeListener(scheduler(), name,
            factorizationbasedConf.getRetrainAfterPreferenceChanges()),label);
      }

      log.info("Added FactorizationBasedRecommender [{}] using [{}] features and [{}] iterations for label: {}",
          new Object[] { name, factorizationbasedConf.getNumberOfFeatures(),
              factorizationbasedConf.getNumberOfIterations() , label});
}
private void createEmptyFactorization(PersistenceStrategy strategy) throws IOException {
	    strategy.maybePersist(new Factorization(new FastByIDMap<Integer>(0), new FastByIDMap<Integer>(0),
	        new double[0][0], new double[0][0]));
	  }
/**
 * Method maps ids into int range
 * @param id
 * @return long
 */
protected long idRemapping(long id){
	  return Math.abs(id % 2147483647);
}

}
