package org.plista.kornakapi.core.training;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.model.DataModel;
import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.config.RecommenderConfig;
import org.plista.kornakapi.web.Components;
import org.quartz.SchedulerException;

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


            String name = "inference_lda";
            Components components = Components.instance();
            LDARecommenderConfig conf = (LDARecommenderConfig) components.getConfiguration().getLDARecommender();
            Path p = new Path(conf.getLDARecommenderModelPath());
            DocumentTopicInferenceTrainer trainer = new DocumentTopicInferenceTrainer(conf, p);

            components.setTrainer(name, trainer);
            components.scheduler().addRecommenderTrainingJob(name);
            try {
                components.scheduler().immediatelyTrainRecommender(name);
            } catch (SchedulerException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }


        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

}
