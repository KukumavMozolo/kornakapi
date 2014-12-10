package org.plista.kornakapi.core.optimizer;

import org.apache.commons.dbcp.BasicDataSource;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.plista.kornakapi.core.config.Configuration;
import org.plista.kornakapi.core.optimizer.ErrorALSWRFactorizer.ErrorFactorization;
import org.plista.kornakapi.core.storage.MySqlSplitableMaxPersistentStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

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
		for(int feature : features) {
            for (double lambda : lambdas) {
                for (double alpha : alphas) {
                    for (int iter : iterations) {
                        double totalError = 0;
                        for (int i = 0; i < 3; i++) {
                            ErrorFactorization factorization = null;
                            try {
                                ErrorALSWRFactorizer factorizer = new ErrorALSWRFactorizer(trainingSets.get(i), testSets.get(i), feature, lambda,
                                        iter, true, alpha, numProcessors);

                                long start = System.currentTimeMillis();
                                factorization = factorizer.factorize();
                                long estimateDuration = System.currentTimeMillis() - start;

                                if (log.isInfoEnabled()) {
                                    log.info("Model trained in {} ms", estimateDuration);
                                }
                                File targetFile = new File("/opt/kornakapi-model/crossvalidation.model");

                                new FilePersistenceStrategy(targetFile).maybePersist(factorization);
                            } catch (Exception e) {
                                throw new IOException(e);
                            }
                            factorization.getError();
                            // 3 measure performance
                            log.info("Error of {} for features {}, alpha {}, lambda {}, fold: {}", new Object[]{factorization.getError(), feature, alpha, lambda, i});
                            totalError += factorization.getError()[iter];
                        }
                        totalError = totalError /3;
                        data.insertPerformance(label, feature, iter, alpha, lambda, totalError);
                    }
                }
            }
        }
        data.close();
	}

}

