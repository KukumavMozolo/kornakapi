
package org.plista.kornakapi.core.training;

import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.clustering.lda.cvb.TopicModel;
import org.apache.mahout.math.*;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class DocumentTopicInferenceTrainer extends AbstractTrainer{
	private LDARecommenderConfig conf;
	org.apache.hadoop.conf.Configuration lconf = new org.apache.hadoop.conf.Configuration(); 
	private FileSystem fs;
	private int modelWeight = 1;

	private Path path;
	private int trainingThreads;
    private String safeKey;
    private SemanticModel semanticModel = null;
    private static final Logger log = LoggerFactory.getLogger(DocumentTopicInferenceTrainer.class);
	

	public DocumentTopicInferenceTrainer(RecommenderConfig conf, Path path) {
		super(conf);
		this.conf = (LDARecommenderConfig)conf;
		try {
			fs = FileSystem.get(lconf);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		this.path = path;
		trainingThreads = this.conf.getInferenceThreats();
        semanticModel = new SemanticModel(path, (LDARecommenderConfig)conf);
        try {
            safeKey = semanticModel.getModelKey();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    /**
     *
     */
    protected void doTrain(File targetFile, DataModel inmemoryData,
                           int numProcessors) throws IOException {
        semanticModel.read();


        FromFileVectorizer vectorizer = new FromFileVectorizer(conf);
        boolean succes = false;
        try {
            succes = vectorizer.doTrain();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            log.info("Inference Failed");
        }
        if(succes){
            inferTopicsForItems();
        }

    }


    /**
     *
     * @param itemid
     * @param item
     */
	private void inferTopics(Path[] validFiles, String itemid, Vector item){
		if(semanticModel.getItemFeatures().containsKey(itemid)){
            if(log.isInfoEnabled()){
                log.info("Item {} is already known.", itemid);
            }
			return;
		}
		try {

			TopicModel model = new TopicModel(lconf, conf.getEta(), conf.getAlpha(), getDictAsArray(), trainingThreads, modelWeight,
                    validFiles);
			 Vector docTopics = new DenseVector(new double[model.getNumTopics()]).assign(1.0/model.getNumTopics());
			 Matrix docTopicModel = new SparseRowMatrix(model.getNumTopics(), item.size());
			 int maxIters = 1000;
		        for(int i = 0; i < maxIters; i++) {
		            model.trainDocTopicModel(item, docTopics, docTopicModel);
		        }
		    model.stop();
            semanticModel.getItemFeatures().put(itemid, docTopics);
            semanticModel.getIndexItem().put(semanticModel.getIndexItem().size() + 1, itemid);
            semanticModel.getItemIndex().put(itemid, semanticModel.getItemIndex().size() + 1);
            if(log.isInfoEnabled()){
                log.info("Inferred new Feature Vector for item: {}", itemid);
            }
		    
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}
	/**
	 * 
	 */
	private void inferTopicsForItems(){
		HashMap<String, Vector> tfVectors = createVectorsFromDir();
		if(tfVectors== null){ //If there are no topics then there is nothing to infere
            if(log.isInfoEnabled()){
                log.info("tfVectors is null, exiting");
            }
			return;
		}
        if(log.isInfoEnabled()){
            log.info("Infereing topics for {} Vectors", tfVectors.size());
        }
        Path[] models = getallModelPaths();
		for(String itemid : tfVectors.keySet()){
			inferTopics(models,itemid, tfVectors.get(itemid));

		}
		try {
            SemanticModel newModel = new SemanticModel(semanticModel.getIndexItem(),semanticModel.getItemIndex(),semanticModel.getItemFeatures(),path,conf);
            newModel.getModelKey();
            newModel.safe(safeKey);
            if(log.isInfoEnabled()){
                log.info("New InferenceModel Created");
            }
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}

    /**
     * creates an array of pathes of the topic files
     * @return
     */
    private Path[] getallModelPaths() {
        File dir = new File(this.conf.getTopicsOutputPath());
        String[] files = dir.list();
        ArrayList<Path> validFiles = new ArrayList<Path>();
        for(String file :files){
            if(file.contains("part-m") && !file.contains(".crc")){
                validFiles.add(new Path(this.conf.getTopicsOutputPath() + file));
            }
        }
        Path[] validFilesPaths = new Path[validFiles.size()];
        validFilesPaths = validFiles.toArray(validFilesPaths);
        return validFilesPaths;
    }

    /**
	 * 
	 * @return Returns Dictionary
	 * @throws IOException
	 */
	private String[] getDictAsArray() throws IOException{
		ArrayList<String> dict = new ArrayList<String>();
        Reader reader = new SequenceFile.Reader(fs,new Path(this.conf.getTopicsDictionaryPath()) , lconf);
        Text keyNewDict = new Text();
        IntWritable newVal = new IntWritable();
        while(reader.next(keyNewDict,newVal)){
            dict.add(keyNewDict.toString());
        }
		Closeables.close(reader, false);
        return dict.toArray(new String[dict.size()]);
	}

    /**
     *
     * @return
     */
	private HashMap<String,Vector> getNewVectors(){
		HashMap<String, Vector> newVectors = new HashMap<String, Vector>();
		try {
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(conf.getInferencePath() + "sparsein/tfidf-vectors/part-r-00000"), lconf);
			Text key = new Text();
			VectorWritable val = new VectorWritable();
			while(reader.next(key, val)){
				newVectors.put(key.toString().substring(1), val.get());
			}
			Closeables.close(reader, false);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return newVectors;
	}

    /**
     *
     * @return
     */
	private HashMap<String, Vector> createVectorsFromDir() {
        HashMap<String, Vector> newVectors = getNewVectors();
		//cleanup(newVectors);
		ArrayList<String> newDict = null;
		HashMap<String,Integer> oldDict = null;
		try {
			newDict = this.getFileDict();
			oldDict = getDict();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        HashMap<String, Vector> tfVectors = new HashMap<String, Vector>();  
		if(oldDict != null){
	        for(String key : newVectors.keySet()){
	        	RandomAccessSparseVector newArticleTF = new RandomAccessSparseVector(oldDict.size());
	        	Vector val = newVectors.get(key);
	        	for(int i = 0; i< val.size(); i++){
					double tf =val.get(i);
					String ngram = newDict.get(i);
					if(oldDict.containsKey(ngram)){
						int idx = oldDict.get(ngram);
						newArticleTF.set(idx, tf);
					}
	        	}
				tfVectors.put(key, newArticleTF);
	        }
		}
		return tfVectors;
	}


    /**
     *
     * @return Dictionary as HashMap word as key, integer id as value
     * @throws IOException
     */
	private HashMap<String,Integer> getDict() throws IOException{
        HashMap<String,Integer> modelDictionary = new HashMap<String, Integer>();
		Reader reader = new SequenceFile.Reader(fs,new Path(this.conf.getTopicsDictionaryPath()) , lconf);
		Text keyModelDict = new Text();
	    IntWritable valModelDict = new IntWritable();
	    while(reader.next(keyModelDict, valModelDict)){
	        modelDictionary.put(keyModelDict.toString(), Integer.parseInt(valModelDict.toString()));
	    }   
		Closeables.close(reader, false);

    	return modelDictionary;

	} 
	
	/*
	 * 
	 */
	private ArrayList<String> getFileDict() throws IOException{
        ArrayList<String> dict = new ArrayList<String>();
        Reader reader = new SequenceFile.Reader(fs, new Path(conf.getInferencePath() + "/sparsein/dictionary.file-0"), lconf);
        Text keyText = new Text();
        IntWritable valCount = new IntWritable();
        while(reader.next(keyText,valCount)){
            dict.add(keyText.toString());
        }
        Closeables.close(reader, false);
        return dict;
	}





    /**
     *
     * @param tfVectors
     */
	protected void cleanup(HashMap<String, Vector> tfVectors){
		for(String key : tfVectors.keySet()){
			File from = new File(conf.getInferencePath() + "Documents/"+ conf.getTrainingSetName() +"/" + key);
			File to = new File(conf.getTextDirectoryPath()+ key);
			try {
				File newFile = new File(to.toString());
				if(newFile.exists()){
					newFile.delete();
				}
				Files.copy(from, to);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}
