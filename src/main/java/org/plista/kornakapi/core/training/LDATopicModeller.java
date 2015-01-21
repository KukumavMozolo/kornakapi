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
import org.plista.kornakapi.core.config.RecommenderConfig;

import java.io.File;
import java.io.IOException;


/**
 * Trainer to train a semantic model using lda
 *
 */
public class LDATopicModeller extends AbstractTrainer{
	protected RecommenderConfig conf;

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
    }
	
}
