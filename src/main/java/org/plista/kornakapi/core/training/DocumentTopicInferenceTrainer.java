
package org.plista.kornakapi.core.training;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.clustering.lda.cvb.TopicModel;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.TFIDF;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;


public class DocumentTopicInferenceTrainer extends AbstractTrainer{
	private LDARecommenderConfig conf;
	org.apache.hadoop.conf.Configuration hadoopConf = new org.apache.hadoop.conf.Configuration();
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
			fs = FileSystem.get(hadoopConf);
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
        if(log.isInfoEnabled()){
            log.info("LDA: Model read, starting to create sequence files");
        }

        FromFileVectorizer vectorizer = new FromFileVectorizer(conf);
        boolean succes = false;
        try {
            succes = vectorizer.doTrain();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            log.info("LDA: Inference Failed");
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
	private void inferTopics( Path[] models, String itemid, Vector item, String[] dict){
		if(semanticModel.getItemFeatures().containsKey(itemid)){
            if(log.isInfoEnabled()){
                log.info("LDA: Item {} is already known.", itemid);
            }
			return;
		}
		try {

            TopicModel model = new TopicModel(hadoopConf, conf.getEta(), conf.getAlpha(), dict, trainingThreads, modelWeight,
                    models);
            if(log.isInfoEnabled()){
                log.info("LDA: Model : {}", models[3].toString());
            }

			 Vector docTopics = new DenseVector(new double[model.getNumTopics()]).assign(1.0/model.getNumTopics());
			 Matrix docTopicModel = new SparseRowMatrix(model.getNumTopics(), item.size());
			 int maxIters = 5000;
		        for(int i = 0; i < maxIters; i++) {
		            model.trainDocTopicModel(item, docTopics, docTopicModel);
		        }
            model.stop();
            semanticModel.getItemFeatures().put(itemid, docTopics);
            semanticModel.getIndexItem().put(semanticModel.getIndexItem().size() + 1, itemid);
            semanticModel.getItemIndex().put(itemid, semanticModel.getItemIndex().size() + 1);
            if(log.isInfoEnabled()){
                log.info("LDA: Inferred new Feature Vector for item: {}, values: {}", itemid, docTopics.toString());
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
                log.info("LDA: tfVectors is null, exiting");
            }
			return;
		}
        if(log.isInfoEnabled()){
            log.info("LDA: Infereing topics for {} Vectors", tfVectors.size());
        }
        Path[] models = getallModelPaths();
		try {
            String[] dict = getDictAsArray();

            for(String itemid : tfVectors.keySet()){
                inferTopics(models, itemid, tfVectors.get(itemid), dict);

            }

            SemanticModel newModel = new SemanticModel(semanticModel.getIndexItem(),semanticModel.getItemIndex(),semanticModel.getItemFeatures(),path,conf);
            newModel.getModelKey();
            newModel.safe(safeKey);
            if(log.isInfoEnabled()){
                log.info("LDA: New InferenceModel Created");
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
        Reader reader = new SequenceFile.Reader(fs,new Path(this.conf.getTopicsDictionaryPath()) , hadoopConf);
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
	private HashMap<String, Vector> createVectorsFromDir() {
        Map<String, Integer> dict = readDictionnary(hadoopConf, new Path(conf.getTopicsDictionaryPath()));
        Map<Integer, Long> dfcounter = readDocumentFrequency(hadoopConf,new Path(conf.getSparseVectorOutputPath() + "df-count/part-r-00000") );
        Map<String, HashMap<String,Integer>> newDocument = getNewDocuments(hadoopConf, new Path(conf.getInferencePath() + "sparsein/tokenized-documents/part-m-00000"));

        HashMap<String, Vector> tfVectors = new HashMap<String, Vector>();
        if(dict!=null && !dict.isEmpty()) {
            TFIDF tfidf = new TFIDF();
            int numberDocs = dfcounter.get(-1).intValue();
            for (Map.Entry<String, HashMap<String, Integer>> doc : newDocument.entrySet()) {
                String itemId = doc.getKey();
                HashMap<String, Integer> doctf = doc.getValue();
                RandomAccessSparseVector docTfIdf = new RandomAccessSparseVector(dict.size());
                for (Map.Entry<String, Integer> n : doctf.entrySet()) {
                    String word = n.getKey();
                    Integer count = n.getValue();
                    if (dict.containsKey(word)) {
                        int idx = dict.get(word);
                        long worddf = dfcounter.get(idx);
                        double idf = tfidf.calculate(count, (int) worddf, 0, numberDocs);
                        docTfIdf.set(idx, idf);
                    }

                }
                if(log.isInfoEnabled()){
                    log.info("LDA: Created vector for: " + itemId +": " + docTfIdf.toString());
                }
                tfVectors.put(itemId, docTfIdf);
            }
        }
        return tfVectors;
	}

    /**
     *
     * @param conf
     * @param dictionnaryPath
     * @return
     */
    private static Map<String, Integer> readDictionnary(Configuration conf, Path dictionnaryPath) {
        Map<String, Integer> dictionnary = new HashMap<String, Integer>();
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(dictionnaryPath, true, conf)) {
            dictionnary.put(pair.getFirst().toString(), pair.getSecond().get());
        }
        return dictionnary;
    }

    /**
     *
     * @param conf
     * @param documentFrequencyPath
     * @return
     */
    private static Map<Integer, Long> readDocumentFrequency(Configuration conf, Path documentFrequencyPath) {
        Map<Integer, Long> documentFrequency = new HashMap<Integer, Long>();
        for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(documentFrequencyPath, true, conf)) {
            documentFrequency.put(pair.getFirst().get(), pair.getSecond().get());
        }
        return documentFrequency;
    }


    /**
     * Creates hashMap: <itemId, HashMap: <Word, wordcount>>
     * @param conf
     * @param tokenizedDocsPath
     * @return
     */
    private static HashMap<String,HashMap<String,Integer>> getNewDocuments(Configuration conf, Path tokenizedDocsPath) {
        HashMap<String,HashMap<String,Integer>> idDocumentTF = new HashMap<String, HashMap<String, Integer>>();
        for (Pair<Text, StringTuple> pair : new SequenceFileIterable<Text, StringTuple>(tokenizedDocsPath, true, conf)) {
            String itemId = pair.getFirst().toString().substring(1);
            List<String> words =pair.getSecond().getEntries();
            List<String> done = new ArrayList<String>();
            HashMap<String, Integer> docVector = new HashMap<String,Integer>();
            for(String word : words){
                if(!done.contains(word)){
                    int count = Collections.frequency(words, word);
                    docVector.put(word,count);
                }
                done.add(word);
            }
            idDocumentTF.put(itemId,docVector);

        }
        return idDocumentTF;
    }


}
