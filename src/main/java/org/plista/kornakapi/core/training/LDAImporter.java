package org.plista.kornakapi.core.training;

import org.apache.mahout.cf.taste.model.DataModel;
import org.plista.kornakapi.core.config.RecommenderConfig;

import java.io.File;
import java.io.IOException;

/**
 *
 */
public class LDAImporter extends LDATrainer {
    public LDAImporter(RecommenderConfig conf) {
        super(conf);
    }


    @Override
    protected void doTrain(File targetFile, DataModel inmemoryData,
                           int numProcessors) throws IOException {
        try {
            new LDATopicModeller(conf).doImport();
            log.info("New Model Imported");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

}
