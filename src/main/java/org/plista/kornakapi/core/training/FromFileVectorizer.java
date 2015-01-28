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

import com.google.common.collect.Lists;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.text.SequenceFilesFromDirectory;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * 
 *
 */
public class FromFileVectorizer {

    private static final Logger log = LoggerFactory.getLogger(FromFileVectorizer.class);
	private Path DocumentFilesPath;
	private Path sequenceFilesPath;
	private Path sparseVectorOut;
	private LDARecommenderConfig conf;
	/**
	 * 
	 * @param conf
	 */
	protected FromFileVectorizer(LDARecommenderConfig conf) {
		DocumentFilesPath = new Path(conf.getTextDirectoryPath() );
		sequenceFilesPath = new Path(conf.getTextDirectoryPath()+ "seq/");
		sparseVectorOut= new Path(conf.getInferencePath() + "sparsein/");
		this.conf = conf;
	}

	protected void doTrain() throws Exception {
		generateSequneceFiles();
		generateSparseVectors(false,true,this.conf.getMaxDFSigma(),sequenceFilesPath,sparseVectorOut);

	}
	/**
	 * Generates SequenceFile
	 * @throws Exception 
	 */
	private void generateSequneceFiles() throws Exception{
        if(log.isInfoEnabled()){
            log.info("From: " + DocumentFilesPath.toString() +" To: " + sequenceFilesPath.toString());
        }
		List<String> argList = Lists.newLinkedList();
        argList.add("-i");
        argList.add(DocumentFilesPath.toString());
        argList.add("-o");
        argList.add(sequenceFilesPath.toString());
        argList.add("-ow");
        String[] args = argList.toArray(new String[argList.size()]);
        ToolRunner.run(new SequenceFilesFromDirectory(), args);

	}
	/**
	 * 
	 * @param tfWeighting, either if true tf(unnormalized term-frequency) else TFIDF(normalized through maxFrequncy)
	 * @param named, if true output Vectors are named
	 * @param maxDFSigma, Maximum Standard deviation of termfrequency, 
	 * @param inputPath
	 * @param outputPath
	 * @throws Exception
	 */
    private void generateSparseVectors (boolean tfWeighting,  boolean named, double maxDFSigma, Path inputPath, Path outputPath) throws Exception {
        List<String> argList = Lists.newLinkedList();
        argList.add("-ow");
        argList.add("-i");
        argList.add(inputPath.toString());
        argList.add("-o");
        argList.add(outputPath.toString());
        argList.add("-seq");
        if (named) {
            argList.add("-nv");
        }
        if (maxDFSigma >= 0) {
            argList.add("--maxDFSigma");
            argList.add(String.valueOf(maxDFSigma));
        }
        if (tfWeighting) {
            argList.add("--weight");
            argList.add("tf");
        }else{
            argList.add("--weight");
            argList.add("tfidf");
        }
        String[] args = argList.toArray(new String[argList.size()]);
        //String[] seqToVectorArgs = {"--weight", "tfidf", "--input", inputPath.toString(), "--output",  outputPath.toString(), "--maxDFPercent", "70", "--maxNGramSize", "2", "--namedVector"};
        ToolRunner.run(new SparseVectorsFromSequenceFiles(), args);
    }
    
}
