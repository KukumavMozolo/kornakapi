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

import com.google.common.io.Closeables;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.security.PrivilegedExceptionAction;
import java.util.HashMap;


/**
 * Controlls the exectution of LDA on YARN cluster
 *
 */
public class LDATopicVectorizer {
	

    private double doc_topic_smoothening = 0.0001;
    private double term_topic_smoothening = 0.0001;
    private int iteration_block_size = 10;
    private float testFraction = 0.1f;
    private int numTrainThreads ;
    private int numUpdateThreads;
    private int maxItersPerDoc = 10;
    private int numReduceTasks = 10;
    private boolean backfillPerplexity = false;
    private int seed = (int) (Math.random() * 302221);

	
    private Path sparseVectorIn;
	private Path topicsOut;
    private LDARecommenderConfig conf;
	private Integer k;
	private Double alpha;
	private Double eta;
	private Double convergenceDelta = 0.0001;
	org.apache.hadoop.conf.Configuration lconf = new org.apache.hadoop.conf.Configuration(); 
	FileSystem fs;
	private HashMap<Integer,String> indexItem = null;
	private HashMap<String,Vector> itemFeatures;
	private HashMap<String,Integer> itemIndex = null;
    private static final Logger log = LoggerFactory.getLogger(LDATopicVectorizer.class);
	
	/**
	 * 
	 * @param conf
	 * @throws TasteException
	 * @throws IOException
	 */
	protected LDATopicVectorizer(RecommenderConfig conf) throws TasteException, IOException {
		this.conf = (LDARecommenderConfig)conf;
		sparseVectorIn= new Path(this.conf.getYarnInputDir());
		topicsOut= new Path(this.conf.getYarnOutputDir());
		k= this.conf.getnumberOfTopics();
        fs = FileSystem.get(lconf);
        this.alpha = this.conf.getAlpha();
        this.eta = this.conf.getEta();
        this.numTrainThreads = this.conf.getTrainingThreats();
        this.numUpdateThreads = this.conf.getTrainingThreats();
        this.numReduceTasks = this.conf.getTrainingThreats();
	}
	
	/**
	 * Method mapping array index to itemIds
	 * @throws IOException
	 */
	private void indexItem() throws IOException{
		if(indexItem == null){
			indexItem = new HashMap<Integer,String>();
			itemIndex = new HashMap<String,Integer>();
			Reader reader = new SequenceFile.Reader(fs,new Path(this.conf.getCVBInputPath() + "/docIndex") , lconf);
			IntWritable key= new IntWritable();
			Text  newVal = new Text();
			while(reader.next(key, newVal)){
				indexItem.put(key.get(),newVal.toString().substring(1));
				itemIndex.put(newVal.toString().substring(1),key.get());
			}
			Closeables.close(reader, false);
		}
	}
	
	/**
	 *
	 * @param itemid
	 * @return item index of itemid
	 * @throws IOException
	 */
	public Integer getitemIndex(String itemid) throws IOException{
		if(itemIndex==null){
			indexItem();
		}
		return itemIndex.get(itemid);
	}
	
	/**
	 * Method mapping array index to itemIds
	 * @param idx
	 * @return Itemid for item index
	 * @throws IOException
	 */
	public String getIndexItem(Integer idx) throws IOException{
		if(indexItem==null){
			indexItem();
		}
		return indexItem.get(idx);
	}
	

	
	/**
	 * gets topic posterior from lda output
	 * @throws IOException
	 */
	private void getAllTopicPosterior() throws IOException{
		itemFeatures= new HashMap<String,Vector>();
		Reader reader = new SequenceFile.Reader(fs,new Path(this.conf.getTopicsOutputPath() + "DocumentTopics/part-m-00000") , lconf);
		IntWritable key = new IntWritable();
		VectorWritable newVal = new VectorWritable();
		while(reader.next(key, newVal)){
			itemFeatures.put(getIndexItem(key.get()), newVal.get());
		}
		Closeables.close(reader, false);		
	}
	
	/**
	 * 
	 * @return
	 * @throws TasteException
	 */
	public SemanticModel vectorize() throws TasteException, IOException {
        /**			//MapReduce
         CVB0Driver driver = new CVB0Driver();
         Configuration jobConf = new Configuration();
         driver.run(jobConf, sparseVectorIn.suffix("/matrix"),
         topicsOut, k, 2000, doc_topic_smoothening, term_topic_smoothening,
         maxIter, iteration_block_size, convergenceDelta,
         new Path(((LDARecommenderConfig)conf).getTopicsDictionaryPath()), new Path(((LDARecommenderConfig)conf).getLDADocTopicsPath()), new Path(((LDARecommenderConfig)conf).getTmpLDAModelPath()),
         seed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc,
         numReduceTasks, backfillPerplexity);
         **/


/**
       	List<String> argList = Lists.newLinkedList();
        argList.add("-i");
        argList.add(sparseVectorIn.toString()+ "/matrix");
        argList.add("-to");
        argList.add(topicsOut.toString() );
        argList.add("--numTopics");
        argList.add(k.toString());
        argList.add("-d");
        argList.add(((LDARecommenderConfig)conf).getTopicsDictionaryPath());
        argList.add("--alpha");
        argList.add(alpha.toString());
        argList.add("--eta");
        argList.add(eta.toString());
        argList.add("-do");
        argList.add(((LDARecommenderConfig)conf).getLDADocTopicsPath());
        argList.add("-c");
        argList.add(convergenceDelta.toString());
        argList.add("-ntt");
        argList.add(((LDARecommenderConfig)conf).getTrainingThreats().toString());
        argList.add("-m");
        argList.add(((LDARecommenderConfig)conf).getMaxIterations());
        argList.add("-nut");
        argList.add(((LDARecommenderConfig)conf).getTrainingThreats().toString());
       String[] args = argList.toArray(new String[argList.size()]);
       /**
        try {
		InMemoryCollapsedVariationalBayes0.main(args);
	    //computeAllTopicPosterior();
		getAllTopicPosterior();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
 **/

        UserGroupInformation ugi = UserGroupInformation.createRemoteUser(conf.getHadoopUser());
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hadoopConf = new Configuration();
                    hadoopConf.addResource(new Path(conf.getHadoopConfPath()));

                    int maxIter =Integer.parseInt(conf.getMaxIterations());
                    CVB0Driver driver = new CVB0Driver();
                    int numTerms = getNumTerms(new Path(conf.getTopicsDictionaryPath()));

                    try {
                        driver.run(hadoopConf, sparseVectorIn.suffix("/matrix"),
                                topicsOut, k, numTerms, doc_topic_smoothening, term_topic_smoothening,
                                maxIter, iteration_block_size, convergenceDelta,
                                sparseVectorIn.suffix("/dictionary.file-0"), topicsOut.suffix("/DocumentTopics/"), sparseVectorIn,
                                seed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc,
                                numReduceTasks, backfillPerplexity);
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    FileSystem fileSystem = FileSystem.get(hadoopConf);

                    Path dest = new Path(conf.getTopicsOutputPath());
                    FileUtils.deleteDirectory(new File(dest.toString()));
                    try {
                        fileSystem.copyToLocalFile(new Path(conf.getYarnOutputDir() ), dest );
                    } catch (IOException e) {
                        e.printStackTrace();

                    }finally {
                        fileSystem.close();
                    }
                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        getAllTopicPosterior();
        return  new SemanticModel(indexItem,itemIndex, itemFeatures, new Path(conf.getLDARecommenderModelPath()),conf);
	}


    /**
     * Method to connect to yarn and download newest topicModel
     * @return SemanticModel
     * @throws IOException
     */
    public SemanticModel getModelFromYarn() throws IOException {

        UserGroupInformation ugi = UserGroupInformation.createRemoteUser(conf.getHadoopUser());
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hadoopConf = new Configuration();
                    hadoopConf.addResource(new Path(conf.getHadoopConfPath()));
                    FileSystem fileSystem = FileSystem.get(hadoopConf);

                    Path dest = new Path(conf.getTopicsOutputPath());
                    FileUtils.deleteDirectory(new File(dest.toString()));
                    try {
                        Path outputDir = new Path(conf.getYarnOutputDir());

                        fileSystem.copyToLocalFile(outputDir, dest );
                        fileSystem.copyToLocalFile(new Path(conf.getYarnOutputDir() + "/docIndex"), new Path(conf.getCVBInputPath() + "/docIndex"));
                        fileSystem.copyToLocalFile(new Path(conf.getYarnOutputDir() + "/dictionary.file-0"),  new Path(conf.getTopicsDictionaryPath()));
                    } catch (IOException e) {
                        e.printStackTrace();

                    }finally {
                        fileSystem.close();
                    }
                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        getAllTopicPosterior();
        return  new SemanticModel(indexItem,itemIndex, itemFeatures, new Path(conf.getLDARecommenderModelPath()),conf);
    }

    /**
     *
     * @param dictionaryPath
     * @return
     * @throws IOException
     */
    private static int getNumTerms( Path dictionaryPath) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = dictionaryPath.getFileSystem(conf);
        Text key = new Text();
        IntWritable value = new IntWritable();
        int maxTermId = 0;
        SequenceFile.Reader reader = new SequenceFile.Reader(fs,dictionaryPath, conf);
        while (reader.next(key, value)) {
            maxTermId++;
        }
        if(log.isInfoEnabled()){
            log.info("LDA: Max Number of terms per topic: " + Integer.toString(maxTermId));
        }
        reader.close();
        return maxTermId;
    }

}
