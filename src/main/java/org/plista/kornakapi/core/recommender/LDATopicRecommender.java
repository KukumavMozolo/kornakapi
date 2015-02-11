package org.plista.kornakapi.core.recommender;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.BooleanUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Vector;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.training.SemanticModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;



public class LDATopicRecommender extends AbstractRecommender implements KornakapiRecommender {
    private static final Logger log = LoggerFactory.getLogger(LDATopicRecommender.class);

	private LDARecommenderConfig conf;
	private SemanticModel model;
	private final RefreshHelper refreshHelper;
	
	public LDATopicRecommender(DataModel dataModel,	CandidateItemsStrategy allUnknownItemsStrategy, LDARecommenderConfig conf) {
		super(dataModel,allUnknownItemsStrategy );
		this.conf = conf;
		model = new SemanticModel(new Path(conf.getLDARecommenderModelPath()), conf);
		try {
			model.read();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		   refreshHelper = new RefreshHelper(new Callable<Object>() {
			      @Override
			      public Object call() throws TasteException, IOException {
			        reloadModel();
			        return null;
			      }
			    });
		    refreshHelper.addDependency(getDataModel());
		    refreshHelper.addDependency(allUnknownItemsStrategy);
			  }



    @Override
    public List<RecommendedItem> recommendToAnonymous(long[] itemIDs,
                                                      int howMany, IDRescorer rescorer) throws TasteException, NoSuchItemException {
        Long itemId = itemIDs[0];
        Vector itemFeature = model.getItemFeatures(itemId.toString());
        PreferenceArray preferences = asPreferences(itemIDs);
        FastIDSet possibleItemIDs =  getAllOtherItems(Long.MIN_VALUE, preferences, false);
        List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer, new SemanticEstimator(itemFeature));
        if(log.isInfoEnabled()){
            log.info("ItemID: {} , ItemFeatures: {}, topItems: {}",new Object[] {itemId,  itemFeature, topItems});
        }
        return topItems;
    }

    private double semanticPreference(Vector currentFeatures, Long itemID){
    	Vector v;
		try {
			v = model.getItemFeatures(itemID.toString());
			return  currentFeatures.dot(v);
		} catch (NoSuchItemException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return (double)0;
    }


	@Override
	public List<RecommendedItem> recommend(long userID, long[] itemIDs,
			int howMany, IDRescorer rescorer) throws TasteException {
		// TODO Auto-generated method stub
		return null;
	}
	
	  private PreferenceArray asPreferences(long[] itemIDs) {
		    PreferenceArray preferences = new BooleanUserPreferenceArray(itemIDs.length);
		    for (int n = 0; n < itemIDs.length; n++) {
		      preferences.setItemID(n, itemIDs[n]);
		    }
		    return preferences;
		  }
	  private final class SemanticEstimator implements TopItems.Estimator<Long> {

		    private final Vector itemFeatures;

		    private SemanticEstimator(Vector itemFeatures) {
		      this.itemFeatures = itemFeatures;
		    }

		    @Override
		    public double estimate(Long itemID) throws TasteException {
		      return semanticPreference(itemFeatures, itemID);
		    }
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
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		refreshHelper.refresh(alreadyRefreshed);
	}
	  
	private void reloadModel() throws IOException{
		model.read();
	}
}
