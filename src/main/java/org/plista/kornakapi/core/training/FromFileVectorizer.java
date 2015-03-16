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

import java.io.File;
import java.io.IOException;
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
		sequenceFilesPath = new Path(conf.getVectorOutputPath());
		sparseVectorOut= new Path(conf.getInferencePath() + "sparsein/");
		this.conf = conf;
	}

	protected boolean doTrain() throws Exception {
        File dir =  new File(DocumentFilesPath.toString());
        if(dir.list().length == 0){
            if(log.isInfoEnabled()){
                log.info("LDA: Directory is empty. There are no new Articles.");
            }
            return false;
        }
        if(log.isInfoEnabled()){
            log.info("LDA: Starting to create Sequence Files");
        }
        lockDir();
		generateSequneceFiles();
        if(log.isInfoEnabled()){
            log.info("LDA: Sequence Files Generated.");
        }
        unlockDir();
        if(log.isInfoEnabled()){
            log.info("LDA: Starting to create Sparse Vectors");
        }
		generateSparseVectors(false,true,this.conf.getMaxDFSigma(),sequenceFilesPath,sparseVectorOut);
        if(log.isInfoEnabled()){
            log.info("LDA: Sparse Vectors Generated.");
        }
        return  true;

	}

    /**
     * simple unlocking of directory. Communicates to all LDAArticleWriter
     */
    private void unlockDir() {
        File f = new File(DocumentFilesPath.toString() + "lock");
        if(f.exists()){
            f.delete();
        }
        if(log.isInfoEnabled()){
            log.info("LDA: Directory unlocked");
        }
    }

    /**
     * simple locking of directory. Communicates to all LDAArticleWriter
     * Directory needs to be locked to ensure that sequencefiles can be generated since
     * SequenceFilesFromDirectory does throw an error if an articles is not fully saved on hd
     */
    private void lockDir() {
        File f = new File(DocumentFilesPath.toString() + "lock");
        try {
            f.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(log.isInfoEnabled()){
            log.info("LDA: Directory locked");
        }
    }
	/**
	 * Generates SequenceFile
	 * @throws Exception 
	 */
	private void generateSequneceFiles() throws Exception{
		List<String> argList = Lists.newLinkedList();
        argList.add("-i");
        argList.add(DocumentFilesPath.toString());
        argList.add("-o");
        argList.add(sequenceFilesPath.toString());
        argList.add("-ow");
        String[] args = argList.toArray(new String[argList.size()]);
        if(log.isInfoEnabled()){
            log.info("-i {}, -o {}",DocumentFilesPath.toString(), sequenceFilesPath.toString() );
        }
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
