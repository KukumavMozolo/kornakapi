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
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.security.PrivilegedExceptionAction;
import java.util.List;

public class LDATrainer extends AbstractTrainer{
    private static final Logger log = LoggerFactory.getLogger(LDATrainer.class);
	static LDARecommenderConfig conf;
	
	public LDATrainer(RecommenderConfig conf){
		super(conf);
		this.conf = (LDARecommenderConfig) conf;
	}

	@Override
	protected void doTrain(File targetFile, DataModel inmemoryData,
			int numProcessors) throws IOException {
		try {
			collectNewArticles();
			new FromDirectoryVectorizer(conf).doTrain();
            exportSequenceFiletoYarm();
			new LDATopicModeller(conf).doTrain();
            importTopicsFromYarn();
			printLocalTopicWordDistribution(conf,((LDARecommenderConfig)conf).getTopicsOutputPath(),((LDARecommenderConfig)conf).getTopicsOutputPath());
			printLocalDocumentTopicDistribution(conf,((LDARecommenderConfig)conf).getLDADocTopicsPath(),((LDARecommenderConfig)conf).getLDADocTopicsPath());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}

    /**
     hdoopConf.set("fs.defaultFS", "hdfs://192.168.2.233:9000");
     hdoopConf.set("yarn.resourcemanager.hostname", "192.168.2.233");
     hdoopConf.set("mapreduce.framework.name", "yarn");
     hdoopConf.set("mapred.framework.name", "yarn");
     hdoopConf.set("mapred.job.tracker", "192.168.2.233:8032");
     hdoopConf.set("dfs.permissions.enabled", "false");
     **/
	protected void exportSequenceFiletoYarm() throws IOException {



        UserGroupInformation ugi = UserGroupInformation.createRemoteUser("mw");
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hdoopConf = new Configuration();
                    hdoopConf.set("fs.defaultFS", "hdfs://192.168.2.233:9000/user/mw");
                    hdoopConf.set("yarn.resourcemanager.hostname", "192.168.2.233");
                    hdoopConf.set("mapreduce.framework.name", "yarn");
                    hdoopConf.set("mapred.framework.name", "yarn");
                    hdoopConf.set("mapred.job.tracker", "192.168.2.233:8032");
                    hdoopConf.set("dfs.permissions.enabled", "false");
                    hdoopConf.set("hadoop.job.ugi", "mw");


                    String srcString = conf.getCVBInputPath();
                    String dstString = conf.getYarnInputDir();
                    log.info(srcString);

                    Path dstDir = new Path(dstString );
                    FileSystem fileSystem = FileSystem.get(hdoopConf);
                    if ((fileSystem.exists(dstDir))) {
                        fileSystem.delete(dstDir,true);
                    }


                    Path src = new Path(srcString + "matrix");
                    Path dst = new Path(dstString + "matrix");

                    String filename = srcString.substring(srcString.lastIndexOf('/') + 1, srcString.length());



                    fileSystem.copyFromLocalFile(src,dst );

                    String dictString = ((LDARecommenderConfig)conf).getTopicsDictionaryPath();
                    Path dict = new Path(dictString);
                    dst = new Path(dstString + "dictionary.file-0");
                    fileSystem.copyFromLocalFile(dict,dst);

                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     hdoopConf.set("fs.defaultFS", "hdfs://192.168.2.233:9000");
     hdoopConf.set("yarn.resourcemanager.hostname", "192.168.2.233");
     hdoopConf.set("mapreduce.framework.name", "yarn");
     hdoopConf.set("mapred.framework.name", "yarn");
     hdoopConf.set("mapred.job.tracker", "192.168.2.233:8032");
     hdoopConf.set("dfs.permissions.enabled", "false");
     **/
    protected void importTopicsFromYarn() throws IOException {
        Configuration hdoopConf = new Configuration();
        hdoopConf.set("fs.defaultFS", "hdfs://192.168.2.233:9000");
        hdoopConf.set("yarn.resourcemanager.hostname", "192.168.2.233");
        hdoopConf.set("mapreduce.framework.name", "yarn");
        hdoopConf.set("mapred.framework.name", "yarn");
        hdoopConf.set("mapred.job.tracker", "192.168.2.233:8032");
        hdoopConf.set("dfs.permissions.enabled", "false");

        this.conf.getCVBInputPath();
        FileSystem fileSystem = FileSystem.get(hdoopConf);
        try {
            fileSystem.copyToLocalFile(new Path(this.conf.getYarnInputDir()), new Path(this.conf.getTopicsOutputPath()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


	/**
	 * copys all new articles to the corpus
	 */
	protected void collectNewArticles(){
		// train a specific training set
		String trainingSet = conf.getTrainingSetName();
		File newDocs = new File(((LDARecommenderConfig)conf).getInferencePath()+ "Documents/" + trainingSet + '/' );
		String corpusDir = ((LDARecommenderConfig)conf).getTextDirectoryPath();
		for(File from: newDocs.listFiles()){
			File to = new File(corpusDir+ from.getName());
			try {
				File newFile = new File(to.toString());
				if(newFile.exists()){
					newFile.delete();
				}
				Files.move(from, to);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}	
	}
	
	/**
	 * Dumps the Topic distributen to file
	 * @param conf
	 * @param input
	 * @param output
	 */
	public static void printLocalTopicWordDistribution(RecommenderConfig conf, String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input);
	        argList.add("-o");
	        argList.add("/opt/kornakapi-model/lda/print/topics.txt");
	        argList.add("--dictionaryType");
	        argList.add("sequencefile");
	        argList.add("-d");
	        argList.add(((LDARecommenderConfig)conf).getTopicsDictionaryPath());
	        argList.add("-sort");
	        argList.add("true");
	        argList.add("-vs");
	        argList.add("100");
	        String[] args = argList.toArray(new String[argList.size()]);
	        try {
				//LDAPrintTopics.main(args);
				VectorDumper.main(args);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	
	/**
	 * 
	 * @param conf
	 * @param input
	 * @param output
	 */
	public static void printLocalDocumentTopicDistribution(RecommenderConfig conf, String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input);
	        argList.add("-o");
	        argList.add("/opt/kornakapi-model/lda/print/DocumentTopics.txt");
	        argList.add("-sort");
	        argList.add("true");
	        argList.add("-vs");
	        argList.add("100");
	        argList.add("-p");
	        argList.add("true");
	        String[] args = argList.toArray(new String[argList.size()]);
	        try {
				//LDAPrintTopics.main(args);
				VectorDumper.main(args);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}	
}
