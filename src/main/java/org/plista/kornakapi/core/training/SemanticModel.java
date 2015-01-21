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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.lang.RandomStringUtils;

import java.io.IOException;
import java.util.HashMap;

/**
 * This class collects all relevant information for semantic similarity measure using lda
 *
 */
public class SemanticModel{
	private HashMap<Integer,String> indexItem;
	private HashMap<String,Integer> itemIndex;
	private HashMap<String, Vector> itemFeatures;
	private Path path;
    private String key;
	org.apache.hadoop.conf.Configuration lconf = new org.apache.hadoop.conf.Configuration(); 
	FileSystem fs;
	  private static final Logger log = LoggerFactory.getLogger(SemanticModel.class);
	/**
	 * 
	 * @param indexItem hashMap key: index Itemid value
	 * @param itemIndex hashMap key: ItemId, index value
	 * @param itemFeatures hashMap key: ItemId, value TopicVector
	 * @param path path of model directory
	 */
	public SemanticModel(HashMap<Integer,String> indexItem, HashMap<String,Integer> itemIndex, HashMap<String, Vector> itemFeatures, Path path, LDARecommenderConfig conf) throws IOException {
		this.indexItem = indexItem;
		this.itemIndex = itemIndex;
		this.itemFeatures = itemFeatures;
		this.path = path;
		try {
			fs = FileSystem.get(lconf);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

    /**
     *
     * @param path
     * @param conf
     */
	public SemanticModel(Path path, LDARecommenderConfig conf){
		this.path = path;
		try {
			fs = FileSystem.get(lconf);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        try {
            readKey();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @return HashMap IndexItem
     */
	public HashMap<Integer,String> getIndexItem(){
		return indexItem;
	}
	/**
	 * 
	 * @return  HashMap ItemIndex
	 */
	public HashMap<String,Integer> getItemIndex(){
		return itemIndex;
	}
	/**
	 * 
	 * @param itemid
	 * @return
	 */
	public Integer getItemIndex(String itemid){
		return itemIndex.get(itemid);
	}
	/**
	 * 
	 * @return
	 */
	public  HashMap<String, Vector> getItemFeatures(){
		return itemFeatures;
	}
	
	/**
	 * 
	 * @param itemid
	 * @return
	 * @throws NoSuchItemException
	 */
	public Vector getItemFeatures(String itemid) throws NoSuchItemException{
		if(itemFeatures.containsKey(itemid)){
			return itemFeatures.get(itemid);
		}
		throw new NoSuchItemException("Item " + itemid+ " does not exist!");		
	}

    public void safeMaster() throws IOException {
        key = RandomStringUtils.random(10);
        writeKey(key);
        safe(key);

    }

/**
 * Method to safe the model
 * @throws IOException
 */
	public void safe(String safeKey) throws IOException{
        /**
         * New Model training changes the key. Inference can only safe the model if its key is still valid. Thus since inference job start and end no new model was calculated
         */
        if(!this.key.equals(safeKey)){
            if(log.isInfoEnabled()){
                log.info("Storing model Failed. Modelkey Changed");
            }
            return;
        }

		if(itemFeatures !=null){
			Path model = path.suffix("/itemFeature.model");
			Writer w = SequenceFile.createWriter(fs,lconf,model, Text.class, VectorWritable.class);
			for(String itemid : itemFeatures.keySet()){
				Text id = new Text();
				VectorWritable val = new VectorWritable();
				id.set(itemid);
				val.set(itemFeatures.get(itemid));
				w.append(id, val);
			}
			Closeables.close(w, false);
		}
		if(indexItem != null){
			Path model = path.suffix("/indexItem.model");
			Writer w = SequenceFile.createWriter(fs,lconf,model, IntWritable.class, Text.class);
			for(Integer itemid : indexItem.keySet()){
				IntWritable key = new IntWritable();
				Text val = new Text();
				key.set(itemid);
				val.set(indexItem.get(itemid));
				w.append(key, val);
			}
			Closeables.close(w, false);
		}
		if(itemIndex !=null){
			Path model = path.suffix("/itemIndex.model");
			Writer w = SequenceFile.createWriter(fs,lconf,model, Text.class, IntWritable.class);
			for(String itemid : itemIndex.keySet()){
				IntWritable val= new IntWritable();
				Text key  = new Text();
				key.set(itemid);
				val.set(itemIndex.get(itemid));
				w.append(key, val);
			}
			Closeables.close(w, false);
		}
		if(log.isInfoEnabled()){
			log.info("LDA Model Safed");
        }
	}

    /**
     * Key is set to handle concurent writes from DocumentTopicInferenceTrainer and LDATrainer
     * @throws IOException
     */
    private void readKey() throws IOException {
        Path keyPath = path.suffix("/key.txt");
        if(fs.exists(keyPath)){
            Reader reader = new SequenceFile.Reader(fs, keyPath , lconf);
            IntWritable key = new IntWritable();
            Text val = new Text();
            reader.next(key,val);
            this.key = val.toString();
            Closeables.close(reader, false);
            }
       }

    public String getModelKey() throws IOException {
        if(this.key == null){
            readKey();
        }
        return this.key;
    }

    /**
     * Key is set to handle concurent writes from DocumentTopicInferenceTrainer and LDATrainer
     * @throws IOException
     */
    private void writeKey(String key) throws IOException {
        Path keyPath = path.suffix("/key.txt");
        Writer w = SequenceFile.createWriter(fs,lconf,keyPath, IntWritable.class, Text.class);
        IntWritable id = new IntWritable();
        Text val = new Text();
        id.set(1);
        val.set(key);
        w.append(id, val);
        Closeables.close(w, false);
    }


	/**
	 * method to load model from squence file
	 * @throws IOException
	 */
	public void read() throws IOException{
		Path indexPath = path.suffix("/indexItem.model");
		if(fs.exists(indexPath)){
			indexItem = new HashMap<Integer,String>();
			Reader reader = new SequenceFile.Reader(fs, indexPath , lconf);
			IntWritable key = new IntWritable();
			Text val = new Text();
			while(reader.next(key,val)){
				indexItem.put(key.get(), val.toString());
			}
			Closeables.close(reader, false);
		}
		
		Path itemIndexPath = path.suffix("/itemIndex.model");
		if(fs.exists(itemIndexPath)){
			itemIndex = new HashMap<String,Integer>();
			Reader reader = new SequenceFile.Reader(fs, itemIndexPath , lconf);
			IntWritable val = new IntWritable();
			Text key= new Text();
			while(reader.next(key,val)){
				itemIndex.put(key.toString(), val.get());
			}
			Closeables.close(reader, false);
		}
		
		Path featurePath = path.suffix("/itemFeature.model"); 
		if(fs.exists(featurePath)){
			Reader reader = new SequenceFile.Reader(fs, featurePath , lconf);
			itemFeatures = new HashMap<String,Vector>();
			Text key = new Text();
			VectorWritable val = new VectorWritable();
			while(reader.next(key,val)){
				itemFeatures.put(key.toString(), val.get());
			}
			Closeables.close(reader, false);
		}
		if(log.isInfoEnabled()){
					log.info("LDA Model Read");
		}

	}
}