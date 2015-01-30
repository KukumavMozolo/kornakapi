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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;


/**
 * Trainer to train a semantic model using lda
 *
 */
public class LDATopicModeller extends AbstractTrainer{
	protected RecommenderConfig conf;
    private static final Logger log = LoggerFactory.getLogger(LDATopicModeller.class);
	protected LDATopicModeller(RecommenderConfig conf) throws IOException {
		super(conf);
		this.conf = conf;

	}

    /**
     *
     * @throws Exception
     */
    protected void doTrain() throws Exception {
        LDATopicVectorizer vectorize = new LDATopicVectorizer(conf);
        SemanticModel semanticModel = vectorize.vectorize();
        semanticModel.safeMaster();

	}
	@Override
	protected void doTrain(File targetFile, DataModel inmemoryData,
			int numProcessors) throws IOException {	
	}
    protected void doImport() throws IOException, TasteException {
        LDATopicVectorizer vectorize = new LDATopicVectorizer(conf);
        SemanticModel semanticModel = vectorize.getModelFromYarn();
        semanticModel.safeMaster();
        int deletes = removeDublicateArticles(semanticModel);

        if(log.isInfoEnabled()){
            log.info("Deleted " + new Integer(deletes).toString() + " dublicated Articles");
        }
    }

    /**
     *
     * @param model
     * @return
     */
    protected int removeDublicateArticles(SemanticModel model){
        File dir = new File(((LDARecommenderConfig)conf).getTextDirectoryPath());
        HashMap itemIndex = model.getItemIndex();
        int deleteCounter = 0;
        for(File file : dir.listFiles()){
            if(itemIndex.containsKey(file.getName())){
                file.delete();
                deleteCounter++;
            }
        }
        return deleteCounter;
    }
	
}
