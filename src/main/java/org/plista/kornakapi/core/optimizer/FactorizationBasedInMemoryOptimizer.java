package org.plista.kornakapi.core.optimizer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.dbcp.BasicDataSource;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.plista.kornakapi.core.config.Configuration;
import org.plista.kornakapi.core.optimizer.ErrorALSWRFactorizer.ErrorFactorization;
import org.plista.kornakapi.core.storage.MySqlSplitableMaxPersistentStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FactorizationBasedInMemoryOptimizer extends AbstractOptimizer{
	
	private static final Logger log = LoggerFactory.getLogger(FactorizationBasedInMemoryOptimizer.class);	
	

	@Override
	public void optimize(File modelDirectory, Configuration conf, int numProcessors,  String label)
		throws IOException {
			
		int seed = 378934;
		
		// 1 Generate split
		
		BasicDataSource dataSource = new BasicDataSource();
		MySqlSplitableMaxPersistentStorage data = new MySqlSplitableMaxPersistentStorage(conf.getStorageConfiguration(), label,dataSource, seed);
		ArrayList<DataModel> trainingSets = new ArrayList<DataModel>();
		ArrayList<DataModel> testSets = new ArrayList<DataModel>();
		
		trainingSets.add(data.trainingData(0));
		trainingSets.add(data.trainingData(1));
		trainingSets.add(data.trainingData(2));
		
		testSets.add(data.testData(0));
		testSets.add(data.testData(1));
		testSets.add(data.testData(2));
		
		ArrayList<Double> alphas = conf.getFactorizationbasedOptimizer().getAlphaRange();
		ArrayList<Double> lambdas = conf.getFactorizationbasedOptimizer().getLambdaRange();
		ArrayList<Integer> features = conf.getFactorizationbasedOptimizer().getFeatureRange();
		ArrayList<Integer> iterations = conf.getFactorizationbasedOptimizer().getIterationRange();
		
	    log.info("Starting Optimization");
		
		// 2 Loop over all hyperparameters
		for(int feature : features){
			for(double lambda : lambdas){
				for(double alpha : alphas){
					for(int iter : iterations){
						double totalError = 0;
						for(int i = 0; i<3; i++){
							ErrorFactorization factorization = null;
							try {
						      ErrorALSWRFactorizer factorizer = new ErrorALSWRFactorizer(trainingSets.get(i), testSets.get(i), feature, lambda,
						    		  iter, true,alpha, numProcessors);
						      
						      long start = System.currentTimeMillis();
						      factorization = factorizer.factorize();
						      long estimateDuration = System.currentTimeMillis() - start;
						      
						      if (log.isInfoEnabled()) {
						    	  log.info("Model trained in {} ms", estimateDuration);
						      }
						      File targetFile = new File("/opt/kornakapi-model/crossvalidation.model");
						
						      new FilePersistenceStrategy(targetFile).maybePersist(factorization);
					    }catch (Exception e) {
					      throw new IOException(e);
					    }
						// 3 measure performance
						    LongPrimitiveIterator userIDsIterator;
						    try {
								userIDsIterator = testSets.get(i).getUserIDs();
					
	
							    while (userIDsIterator.hasNext()) {
							    	Long userID = userIDsIterator.next();
							    	double[] userf = null;
							    	PreferenceArray userPrefs = testSets.get(i).getPreferencesFromUser(userID);
							    	try{
							    		userf = factorization.getUserFeatures(userID);
							    	} catch(TasteException e){
							    		
							    	}
							    	if(userf != null){
								    	long[] itemIDs = userPrefs.getIDs();
								    	Vector userfVector = new DenseVector(userf);
								    	int idx = 0;
								    	for(long itemID: itemIDs ){
								    		double[] itemf = null;
								    		try{
								    			itemf = factorization.getItemFeatures(itemID);
								    		}catch(TasteException e){
								    			
								    		}
								    		if(itemf != null){
									    		Vector itemfVector = new DenseVector(itemf);
									    		double pref = itemfVector.dot(userfVector);
									    		double realpref = userPrefs.getValue(idx);
									    		idx++;
									    		totalError = totalError + Math.abs(pref - realpref); 
								    		}
	   		  
								    	}							    		
							    	}
	  
							    }
							    log.info("Error of {} for features {}, alpha {}, lambda {}, fold: {}", new Object[]{totalError, feature, alpha, lambda, i});
						    
						} catch (TasteException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						    for(int j = 0; j <factorization.getError().length; j++){
						    	data.insertPerformance(label, feature, j, alpha, lambda, factorization.getError()[j]);
						    }
						    
						}
						data.insertPerformance(label, feature, -1, alpha, lambda, totalError);	
					}							
				}
			}	
		}
		data.close();
	}	
}
