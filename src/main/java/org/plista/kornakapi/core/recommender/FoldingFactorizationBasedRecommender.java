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

package org.plista.kornakapi.core.recommender;

import com.google.common.base.Preconditions;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.BooleanUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.plista.kornakapi.KornakapiRecommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;

/** a matrix factorization based recommender that supports folding in new users */
public final class FoldingFactorizationBasedRecommender extends AbstractRecommender implements KornakapiRecommender {

  private FoldingFactorization foldingFactorization;
  private final PersistenceStrategy persistenceStrategy;
  private final RefreshHelper refreshHelper;
  private final int numEstimationThreads;

  private static final Logger log = LoggerFactory.getLogger(FoldingFactorizationBasedRecommender.class);

  /*public FoldingFactorizationBasedRecommender(DataModel dataModel, CandidateItemsStrategy candidateItemsStrategy,
                                              PersistenceStrategy persistenceStrategy) throws TasteException {
    this(dataModel, candidateItemsStrategy, persistenceStrategy, false);
  }*/

  public FoldingFactorizationBasedRecommender(DataModel dataModel, CandidateItemsStrategy candidateItemsStrategy,
      PersistenceStrategy persistenceStrategy, int numEstimationThreads) throws TasteException {
    super(dataModel, candidateItemsStrategy);

    this.persistenceStrategy = Preconditions.checkNotNull(persistenceStrategy);
    try {
      Factorization factorization = persistenceStrategy.load();
      Preconditions.checkNotNull(factorization, "PersistenceStrategy must provide an initial factorization");
      foldingFactorization = new FoldingFactorization(factorization);
    } catch (IOException e) {
      throw new TasteException("Error loading factorization", e);
    }
    this.numEstimationThreads = numEstimationThreads;

    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        reloadFactorization();
        return null;
      }
    });
    refreshHelper.addDependency(getDataModel());
    refreshHelper.addDependency(candidateItemsStrategy);
  }

  private void reloadFactorization() throws TasteException {
    try {
      Factorization factorization = Preconditions.checkNotNull(persistenceStrategy.load());
      foldingFactorization = new FoldingFactorization(factorization);
    } catch (IOException e) {
      throw new TasteException("Error reloading factorization", e);
    }
  }
  
	@Override
	public List<RecommendedItem> recommend(long userID, int howMany,
			IDRescorer rescorer) throws TasteException {
		
		long fetchHistoryStart = System.currentTimeMillis();
		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		long fetchHistoryDuration = System.currentTimeMillis() - fetchHistoryStart;
		
		if (log.isDebugEnabled()) {
		    	log.debug("fetched {} interactions of user {} in {} ms", 
		    			new Object[] {preferencesFromUser.length(), userID, fetchHistoryDuration} );
		}
		return recommend(userID, preferencesFromUser.getIDs(), howMany, rescorer);
	}

    @Override
    public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems) throws TasteException {
        return null;
    }

    @Override
  public List<RecommendedItem> recommend(long userID, long[] itemIDs, int howMany, IDRescorer rescorer) throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);
    
    PreferenceArray preferencesFromUser = asPreferences(itemIDs);
      
    long fetchItemIDsStart = System.currentTimeMillis();
    FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser,false);
    long fetchItemIDsDuration = System.currentTimeMillis() - fetchItemIDsStart;

    long estimateStart = System.currentTimeMillis();
    List<RecommendedItem> topItems;
    if (numEstimationThreads > 1) {
      topItems = ParallelTopItems.getTopItems(howMany, numEstimationThreads, possibleItemIDs, rescorer,
                                              new Estimator(userID));
    } else {
      topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer, new Estimator(userID));
    }
    long estimateDuration = System.currentTimeMillis() - estimateStart;
    
    long numCandidates = -1;
    if (rescorer != null && rescorer instanceof FixedCandidatesIDRescorer) {
      numCandidates = ((FixedCandidatesIDRescorer) rescorer).numCandidates();
    }
    
    if (log.isDebugEnabled()) {
    	log.debug("fetched {} interactions of user {} ({} itemIDs in {} ms, estimation of {} in {} ms)", 
    			new Object[] { preferencesFromUser.length(), userID, possibleItemIDs.size(), fetchItemIDsDuration, numCandidates, estimateDuration });

    }
    
    log.debug("Recommendations are: {}", topItems);

    return topItems;
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException, NoSuchItemException, NoSuchUserException {
	  double[] userFeatures = foldingFactorization.factorization().getUserFeatures(userID);
	  double[] itemFeatures = foldingFactorization.factorization().getItemFeatures(itemID);
	  return (float) dotProduct(userFeatures, itemFeatures);
  }

  private float estimatePreferenceForAnonymous(double[] foldedInUserFeatures, long itemID) throws NoSuchItemException {
	double[] itemFeatures = foldingFactorization.factorization().getItemFeatures(itemID);
    return (float) dotProduct(foldedInUserFeatures, itemFeatures);
  }

  private double dotProduct(double[] userFeatures, double[] itemFeatures) {
    double dot = 0;
    for (int feature = 0; feature < userFeatures.length; feature++) {
      dot += userFeatures[feature] * itemFeatures[feature];
    }
    return dot;
  }

  @Override
  public List<RecommendedItem> recommendToAnonymous(long[] itemIDs, int howMany, IDRescorer rescorer)
      throws TasteException {

    //TODO what to do here in the non-implicit case? choose a rating?
	long fetchItemIDsStart = System.currentTimeMillis();
    PreferenceArray preferences = asPreferences(itemIDs);
    long fetchItemIDsDuration = System.currentTimeMillis() - fetchItemIDsStart;

    long estimateStart = System.currentTimeMillis();
    double[] foldedInUserFeatures = foldingFactorization.foldInAnonymousUser(itemIDs);

    FastIDSet possibleItemIDs = getAllOtherItems(Long.MIN_VALUE, preferences,false);
    List<RecommendedItem> topItems;
 
    if (numEstimationThreads > 1) {
        topItems = ParallelTopItems.getTopItems(howMany, numEstimationThreads, possibleItemIDs, rescorer,
        		new AnonymousEstimator(foldedInUserFeatures));
      } else {
        topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer, new AnonymousEstimator(foldedInUserFeatures));
      }
    log.debug("Recommendations are: {}", topItems);
    long estimateDuration = System.currentTimeMillis() - estimateStart;
    if (log.isDebugEnabled()) {
    	log.debug("fetched {} interactions of Anonymous ({} itemIDs in {} ms, estimation in {} ms)", 
    			new Object[] { preferences.length(), preferences.length(), fetchItemIDsDuration, estimateDuration });

    }
    return topItems;
  }

  private PreferenceArray asPreferences(long[] itemIDs) {
    PreferenceArray preferences = new BooleanUserPreferenceArray(itemIDs.length);
    for (int n = 0; n < itemIDs.length; n++) {
      preferences.setItemID(n, itemIDs[n]);
    }
    return preferences;
  }

  private final class Estimator implements TopItems.Estimator<Long> {

    private final long theUserID;

    private Estimator(long theUserID) {
      this.theUserID = theUserID;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return estimatePreference(theUserID, itemID);
    }
  }




  private final class AnonymousEstimator implements TopItems.Estimator<Long> {

    private final double[] foldedInUserFeatures;

    private AnonymousEstimator(double[] foldedInUserFeatures) {
      this.foldedInUserFeatures = foldedInUserFeatures;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return estimatePreferenceForAnonymous(foldedInUserFeatures, itemID);
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }


}
