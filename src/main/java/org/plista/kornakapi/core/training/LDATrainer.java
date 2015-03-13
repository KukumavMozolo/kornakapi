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
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.security.PrivilegedExceptionAction;
import java.util.HashMap;
import java.util.List;

/**
 * Trainer for the LDA model. Maintains database. Retrains and downloads the model if the server is LDA Master Server or downloads the model if the server is a LDA slave server.
 * Prints out the topics and Documunet Topic distribution.
 */
public class LDATrainer extends AbstractTrainer{
    protected static final Logger log = LoggerFactory.getLogger(LDATrainer.class);
	static LDARecommenderConfig conf;
	
	public LDATrainer(RecommenderConfig conf){
		super(conf);
		this.conf = (LDARecommenderConfig) conf;
	}

	@Override
	protected void doTrain(File targetFile, DataModel inmemoryData,
			int numProcessors) throws IOException {
		try {

			new FromDirectoryVectorizer(conf).doTrain();
            log.info("LDA: TFIDF - Sequence Files generated");
            exportSequenceFiletoYarm();
            log.info("LDA: TFIDF - Sequence Files uploaded to Cluster");
            deleteOldModelOnYarn();
			new LDATopicModeller(conf).doTrain();
            copyRelevantFiles();
            log.info("LDA: New Model Trained");
			printTopicWordDistribution(conf, conf.getTopicsOutputPath(), conf.getLdaPrintPath());
            log.info("LDA: Topics Printed to " +  conf.getLdaPrintPath());
            DocumentTopicsPrinter();
            log.info("LDA: Document Topics printed to "+  conf.getLdaPrintPath());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}

    /**
     *
     * @throws IOException
     */
    protected void copyRelevantFiles() throws IOException {
        UserGroupInformation ugi = UserGroupInformation.createRemoteUser(conf.getHadoopUser());
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hadoopConf = new Configuration(false);
                    hadoopConf.addResource(new Path(conf.getHadoopConfPath()));
                    Path inputDir = new Path(conf.getYarnInputDir() );
                    FileSystem fileSystem = FileSystem.get(hadoopConf);
                    if (fileSystem.exists(inputDir)) {
                        FileUtil.copy(fileSystem,new Path(conf.getYarnInputDir() + "/docIndex"),fileSystem,new Path(conf.getYarnOutputDir() + "/docIndex" ),false,hadoopConf);
                        FileUtil.copy(fileSystem,new Path(conf.getYarnInputDir() + "/dictionary.file-0"),fileSystem,new Path(conf.getYarnOutputDir() +  "/dictionary.file-0" ),false,hadoopConf);

                    }
                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    protected void exportSequenceFiletoYarm() throws IOException {
        UserGroupInformation ugi = UserGroupInformation.createRemoteUser(conf.getHadoopUser());
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hadoopConf = new Configuration(false);
                    hadoopConf.addResource(new Path(conf.getHadoopConfPath()));

                    String srcString = conf.getCVBInputPath();
                    String dstString = conf.getYarnInputDir();

                    Path dstDir = new Path(dstString );
                    FileSystem fileSystem = FileSystem.get(hadoopConf);
                    if ((fileSystem.exists(dstDir))) {
                        fileSystem.delete(dstDir,true);
                    }
                    Path src = new Path(srcString + "matrix");
                    Path dst = new Path(dstString + "matrix");
                    fileSystem.copyFromLocalFile(src,dst );
                    String dictString = conf.getTopicsDictionaryPath();
                    Path dict = new Path(dictString);
                    dst = new Path(dstString + "dictionary.file-0");
                    fileSystem.copyFromLocalFile(dict,dst);

                    src = new Path(conf.getCVBInputPath() + "/docIndex");
                    dst = new Path(dstString + "docIndex");
                    fileSystem.copyFromLocalFile(src,dst);

                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @throws IOException
     */
    protected void deleteOldModelOnYarn() throws IOException {
        UserGroupInformation ugi = UserGroupInformation.createRemoteUser(conf.getHadoopUser());
        try {
            ugi.doAs(new PrivilegedExceptionAction<Void>() {
                public Void run() throws Exception {
                    Configuration hadoopConf = new Configuration(false);
                    hadoopConf.addResource(new Path(conf.getHadoopConfPath()));


                    String outputSting = conf.getYarnOutputDir();
                    Path outputDir = new Path(outputSting );
                    FileSystem fileSystem = FileSystem.get(hadoopConf);
                    if (fileSystem.exists(outputDir)) {
                        int idx = outputDir.toString().indexOf("out");
                        Path oldModel = new Path(outputDir.toString().substring(0, idx) + "old");
                        if(fileSystem.exists(oldModel)){
                            fileSystem.delete(oldModel);
                        }
                        fileSystem.rename(outputDir, oldModel);
                    }
                    return null;
                }
            });
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }



	/**
	 * copys all new articles to the corpus
	 */
	protected void collectNewArticles(){
		// train a specific training set
		String trainingSet = conf.getTrainingSetName();
		File newDocs = new File(conf.getInferencePath()+ "Documents/" + trainingSet + '/' );
		String corpusDir = conf.getTextDirectoryPath();
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
     *
     * @param conf
     * @param input
     * @param output
     */
    public static void printTopicWordDistribution(RecommenderConfig conf, String input, String output){
        File localDir = new File(input);
        File target = new File(output + "topics.txt");
        if(target.exists()){
            target.delete();
        }
        for (final File fileEntry : localDir.listFiles()) {
            if(fileEntry.getName().contains("part") && !fileEntry.getName().contains("crc")){
                List<String> argList = Lists.newLinkedList();
                argList.add("-i");
                argList.add(fileEntry.getPath());
                argList.add("-o");
                argList.add(output + "/tmp.txt");
                argList.add("--dictionaryType");
                argList.add("sequencefile");
                argList.add("-d");
                argList.add(((LDARecommenderConfig)conf).getTopicsDictionaryPath());
                argList.add("-sort");
                argList.add("true");
                argList.add("-p");
                argList.add("true");

                String[] args = argList.toArray(new String[argList.size()]);
                try {
                    //LDAPrintTopics.main(args);
                    VectorDumper.main(args);
                    //append to file
                    String from = output + "/tmp.txt";
                    File fromTemp = new File(from);
                    BufferedReader br = new BufferedReader(new FileReader(fromTemp));
                    String line = br.readLine();
                    StringBuilder sb = new StringBuilder();
                    while (line != null) {
                        sb.append(line);
                        sb.append("\n");
                        line = br.readLine();
                    }
                    String filename= output + "topics.txt";
                    FileWriter fw = new FileWriter(filename,true); //the true will append the new data
                    fw.write(sb.toString());
                    fw.close();
                    br.close();
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }


    }

    public void DocumentTopicsPrinter() throws IOException {
        org.apache.hadoop.conf.Configuration lconf = new org.apache.hadoop.conf.Configuration();
        FileSystem fs = FileSystem.get(lconf);

        HashMap<Integer,String> indexItemid  = new HashMap<Integer, String>();
        SequenceFile.Reader reader = new SequenceFile.Reader(fs,new Path(conf.getCVBInputPath() + "docIndex") , lconf);
        SequenceFile.Writer writer = new SequenceFile.Writer(fs,lconf,new Path(conf.getCVBInputPath() + "docIndexText"), Text.class, IntWritable.class);
        IntWritable idx = new IntWritable();
        Text itemid = new Text();

        while(reader.next(idx,itemid)) {
            indexItemid.put(idx.get(),itemid.toString().substring(1));
        }
        Closeables.close(reader, false);

        reader =  new SequenceFile.Reader(fs,new Path(conf.getTopicsOutputPath() +"DocumentTopics/part-m-00000" ) , lconf);
        VectorWritable vector = new VectorWritable();

        File f = new File(conf.getLdaPrintPath()+"DocumentTopics.txt" );
        if(f.exists()){
            f.delete();
        }
        BufferedWriter output = new BufferedWriter(new FileWriter(f));


        while(reader.next(idx,vector)){
            output.append(indexItemid.get(idx.get()) +": " + vector.toString());
            output.newLine();
        }
        output.close();
        Closeables.close(reader, false);
        Closeables.close(writer, false);
    }





	/**
	 * Dumps the Topic distributen to file
	 * @param input
	 * @param output
	 */
	public static void printLocalTopicWordDistribution( String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input + "/part-m-00000");
	        argList.add("-o");
	        argList.add(output + "topics.txt");
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
	 * @param input
	 * @param output
	 */
	public static void printLocalDocumentTopicDistribution( String input, String output){
	       List<String> argList = Lists.newLinkedList();
	        argList.add("-i");
	        argList.add(input);
	        argList.add("-o");
	        argList.add(output + "DocumentTopics.txt");
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
