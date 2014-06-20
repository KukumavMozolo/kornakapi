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

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.recommender.FixedCandidatesIDRescorer;
import org.plista.kornakapi.web.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/** servlet to request recommendations */
public class RecommendServlet extends BaseServlet {

  private static final Logger log = LoggerFactory.getLogger(RecommendServlet.class);

  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {


    int howMany = getParameterAsInt(request, Parameters.HOW_MANY, Parameters.DEFAULT_HOW_MANY);

    IDRescorer rescorer = null;

	String label = getParameter(request, Parameters.LABEL, false);
	String recommenderName = getParameter(request, Parameters.RECOMMENDER, true) + "_" + label;
    if(!containsTrainer(recommenderName)){
        if (log.isInfoEnabled()) {
            log.info("No recommender assigned for label {}", label);
         }
    	try {
			createRecommenderForLabel(label);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	FastIDSet candidates = storages().get(label).getCandidates(label);
	rescorer = new FixedCandidatesIDRescorer(candidates);
    

    KornakapiRecommender recommender = recommender(recommenderName);

    try {
      List<RecommendedItem> recommendedItems;

      if (hasParameter(request, Parameters.USER_ID)) {
        long userID = getParameterAsLong(request, Parameters.USER_ID, false);
        long duration = 0;
        long[] itemIDs = null;
        if (hasParameter(request, Parameters.ITEM_IDS)) {
            itemIDs = getParameterAsLongArray(request, Parameters.ITEM_IDS);
        }
        if(itemIDs.length == 1 && itemIDs[0] == 0){       	//if therre are no seen itemids provided
            long start = System.currentTimeMillis();
            recommendedItems = recommender.recommend(userID, howMany, rescorer);
            duration = System.currentTimeMillis() - start;
        }else{
            long start = System.currentTimeMillis();
            recommendedItems = recommender.recommend(userID, itemIDs, howMany, rescorer);
            duration = System.currentTimeMillis() - start;
        }
          
        if (log.isInfoEnabled()) {
          log.info("{} recommendations for user {} in {} ms", new Object[] { recommendedItems.size(), userID, duration });
        }
      } else if (hasParameter(request, Parameters.ITEM_IDS)) {
        long[] itemIDs = getParameterAsLongArray(request, Parameters.ITEM_IDS);

        long start = System.currentTimeMillis();
        recommendedItems = recommender.recommendToAnonymous(itemIDs, howMany, rescorer);
        long duration = System.currentTimeMillis() - start;
        
        if (log.isInfoEnabled()) {
          log.info("{} recommendations for anonymous user {} in {} ms", new Object[] {recommendedItems.size(), itemIDs[0], duration});
        }
      } else {
        throw new IllegalStateException("Parameter [" + Parameters.USER_ID + "] or [" + Parameters.ITEM_IDS + "] " +
            "must be supplied!");
      }

      PrintWriter writer = response.getWriter();

      response.setContentType("application/json");

      String separator = "";
      writer.write("[");
      for (RecommendedItem recommendedItem : recommendedItems) {
        writer.write(separator);
        writer.write("{itemID:");
        writer.write(String.valueOf(recommendedItem.getItemID()));
        writer.write(",value:");
        writer.write(String.valueOf(recommendedItem.getValue()));
        writer.write("}");
        separator = ",";
      }
      writer.write("]");

    } catch (NoSuchUserException e) {
	    if (log.isInfoEnabled()) {
	        log.info("Unkown User: {}", e.getMessage());
	    }
    } catch (NoSuchItemException e) {
	    if (log.isInfoEnabled()) {
	        log.info("Unknown Item: {}", e.getMessage());
	    }
    }catch (TasteException e) {
	    if (log.isInfoEnabled()) {
	        log.info("Unknown Item: {}", e.getMessage());
	    }
        throw new ServletException(e);
      }
  }
}
