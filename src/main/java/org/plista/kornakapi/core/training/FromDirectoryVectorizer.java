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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.text.SequenceFilesFromDirectory;
import org.apache.mahout.utils.vectors.RowIdJob;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;

import java.util.List;

public class FromDirectoryVectorizer {
	
	
	private Path DocumentFilesPath;
	private Path sequenceFilesPath;
	private Path sparseVectorOut;
	private Path sparseVectorInputPath;
	private LDARecommenderConfig conf;
	/**
	 * 
	 * @param conf
	 */
	protected FromDirectoryVectorizer(RecommenderConfig conf) {
		this.conf = (LDARecommenderConfig)conf;
		DocumentFilesPath = new Path(this.conf.getTextDirectoryPath());
		sequenceFilesPath = new Path(this.conf.getVectorOutputPath());
		sparseVectorOut= new Path(this.conf.getSparseVectorOutputPath());	
		sparseVectorInputPath = new Path(this.conf.getCVBInputPath());
			

	}

	protected void doTrain() throws Exception {
		generateSequneceFiles();
		generateSparseVectors(false,true,this.conf.getMaxDFSigma(),sequenceFilesPath,sparseVectorOut);
		ensureIntegerKeys(sparseVectorOut.suffix("/tfidf-vectors/part-r-00000"),sparseVectorInputPath);

	}
	
	private void generateSequneceFiles(){
		List<String> argList = Lists.newLinkedList();
        argList.add("-i");
        argList.add(DocumentFilesPath.toString());
        argList.add("-o");
        argList.add(sequenceFilesPath.toString());
        argList.add("-ow");
        String[] args = argList.toArray(new String[argList.size()]);
        try {
			ToolRunner.run(new SequenceFilesFromDirectory(), args);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
        Configuration hdoopConf = new Configuration();


        List<String> argList = Lists.newLinkedList();
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
        ToolRunner.run(hdoopConf, new SparseVectorsFromSequenceFiles(), args);
    }
    
    /**
     * 
     * @param inputPath
     * @param outputPath
     */
	private void ensureIntegerKeys(Path inputPath, Path outputPath){
        List<String> argList = Lists.newLinkedList();
        argList.add("-i");
        argList.add(inputPath.toString());
        argList.add("-o");
        argList.add(outputPath.toString());
        String[] args = argList.toArray(new String[argList.size()]);
    	RowIdJob asd  = new RowIdJob();
    	try {
			asd.run(args);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

}


