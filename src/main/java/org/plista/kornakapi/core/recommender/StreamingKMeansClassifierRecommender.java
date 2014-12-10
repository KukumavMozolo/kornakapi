package org.plista.kornakapi.core.recommender;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.WeightedVector;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.cluster.StreamingKMeansClassifierModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

public class StreamingKMeansClassifierRecommender implements KornakapiRecommender{
	

	private static final Logger log = LoggerFactory.getLogger(StreamingKMeansClassifierRecommender.class);
	private StreamingKMeansClassifierModel model; 
	
	public StreamingKMeansClassifierRecommender( StreamingKMeansClassifierModel model ){
		this.model = model;
	}
	

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany)
			throws TasteException {
		// TODO Auto-generated method stub
		return null;
	}

    @Override
    public List<RecommendedItem> recommend(long userID, int howMany, boolean includeKnownItems) throws TasteException {
        return null;
    }

    @Override
	public List<RecommendedItem> recommend(long userID, int howMany,
			IDRescorer rescorer) throws TasteException {
		// TODO Auto-generated method stub
		return null;
	}

    @Override
    public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems) throws TasteException {
        return null;
    }

    @Override
	public float estimatePreference(long userID, long itemID)
			throws TasteException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setPreference(long userID, long itemID, float value)
			throws TasteException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void removePreference(long userID, long itemID)
			throws TasteException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public DataModel getDataModel() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public List<RecommendedItem>  recommendToAnonymous(long[] itemIDs,
			int howMany, IDRescorer rescorer) throws TasteException {

		List<RecommendedItem> result = Lists.newArrayListWithCapacity(itemIDs.length);
		if(model.getCentroids() == null){
			log.info("No centroids computed");
		}else{
			for(long itemID : itemIDs){
				WeightedVector centroid;
				try {					
					centroid = (WeightedVector)model.getClossestCentroid(itemID).getValue();
					float normWeight = ((float)centroid.getWeight()/(float)model.getMaxWeight()) ;
					/**
					 * TODO: new Version of mahout is supposed to allow acces on the centroid as a vector
					 * then volume should be normalized by volume*((Vector)centroid).getNumNonZeroElements/(float)this.model.getMeanVolume() or getMaxVolume
					 */
					GenericRecommendedItem item = new GenericRecommendedItem(itemID,  normWeight);
					result.add(item);
				} catch (IOException e){
				    if (log.isInfoEnabled()) {
				    	log.info("{}",  e.getMessage()); 			    			
				    }
				}
			}
		}
		if(result.size() == 0){
			throw new NoSuchItemException("Item unkown");
		}
		return result;
	}


	@Override
	public List<RecommendedItem> recommend(long userID, long[] itemIDs,
			int howMany, IDRescorer rescorer) throws TasteException {
		// TODO Auto-generated method stub
		return null;
	}
}
