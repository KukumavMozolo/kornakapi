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

import org.apache.mahout.cf.taste.common.TasteException;
import org.plista.kornakapi.web.Parameters;
import org.quartz.SchedulerException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;


/** servlet to manually trigger the training for a recommender */
public class TrainServlet extends BaseServlet {
	
  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

	String label = getParameter(request, Parameters.LABEL, true);
    String recommenderName = getParameter(request, Parameters.RECOMMENDER, true)+"_"+ label;
    if(getParameter(request, Parameters.RECOMMENDER, true).equals("lda")){
    	recommenderName = "lda";
    }
    if(!containsTrainer(recommenderName)){
    	try {
			createRecommenderForLabel(label);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	try {
		scheduler().addRecommenderTrainingJob(recommenderName);
		scheduler().immediatelyTrainRecommender(recommenderName);
	} catch (SchedulerException e) {
		e.printStackTrace();
	}
  }
}