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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/** a {@link CandidateItemsStrategy} that will cache all itemIDs in memory */
public class CachingAllUnknownItemsCandidateItemsStrategy implements CandidateItemsStrategy {

  private DataModel dataModel;
  private FastIDSet allItemIDs;

  private static final Logger log = LoggerFactory.getLogger(CachingAllUnknownItemsCandidateItemsStrategy.class);

  public CachingAllUnknownItemsCandidateItemsStrategy(DataModel dataModel) throws TasteException {
    this.dataModel = dataModel;
    allItemIDs = loadAllItemIDs(dataModel);
  }


  public FastIDSet getCandidateItems(long userID, PreferenceArray preferencesFromUser, DataModel dataModel)
      throws TasteException {
    FastIDSet possibleItemIDs = allItemIDs.clone();
      if(log.isInfoEnabled()){
          log.info("PossibleItemIDs: {}", possibleItemIDs.toString());
      }
    possibleItemIDs.removeAll(preferencesFromUser.getIDs());
    if(log.isInfoEnabled()){
        log.info("PossibleItemIDsRemoved: {}", possibleItemIDs.toString());
    }
    return possibleItemIDs;
  }

  private FastIDSet loadAllItemIDs(DataModel dataModel) throws TasteException {

    int numItems = dataModel.getNumItems();

    log.info("Loading {} itemIDs into memory", numItems);

    FastIDSet collectedItemIDs = new FastIDSet(dataModel.getNumItems());
    LongPrimitiveIterator allItemIDsIterator = dataModel.getItemIDs();
    while (allItemIDsIterator.hasNext()) {
      collectedItemIDs.add(allItemIDsIterator.next());
    }
    return collectedItemIDs;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    try {
      allItemIDs = loadAllItemIDs(dataModel);
    } catch (TasteException e) {
      throw new RuntimeException("Unable to reload itemIDs from DataModel", e);
    }
  }

    @Override
    public FastIDSet getCandidateItems(long userID, PreferenceArray preferencesFromUser, DataModel dataModel, boolean includeKnownItems) throws TasteException {
        return null;
    }
}