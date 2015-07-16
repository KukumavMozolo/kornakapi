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

import org.plista.kornakapi.web.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/** servlet to add items to a candidate set */
public class AddCandidateServlet extends BaseServlet {
	
	private static final Logger log = LoggerFactory.getLogger(AddCandidateServlet.class);
    String[] list = {"35533"};
    protected List<String> blacklist  = Arrays.asList(list);

    @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    String label = getParameter(request, Parameters.LABEL, true);
    if(blacklist.contains(label)){
        return;
    }


    long itemID = getParameterAsLong(request, Parameters.ITEM_ID, true);
    if(itemID < 0 || itemID > 2147483647){
    	itemID = this.idRemapping(itemID);
    }
    if(label.equals("1477") && log.isInfoEnabled()){
        log.info("Adding {} to label {}", itemID, label);
    }
    try{
        this.storages().get(label).addCandidate(label, itemID);
    } catch(NullPointerException e){
        if(log.isInfoEnabled()){
            log.info("No Recommender found for label {} and itemID {}", label, itemID );
        }
    }
  }
}
