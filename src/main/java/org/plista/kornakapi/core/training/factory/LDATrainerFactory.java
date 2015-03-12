package org.plista.kornakapi.core.training.factory;

import org.plista.kornakapi.core.config.LDARecommenderConfig;
import org.plista.kornakapi.core.training.LDAImporter;
import org.plista.kornakapi.core.training.LDATrainer;

/**
 * Created by mw on 1/21/15.
 */
public class LDATrainerFactory {

    private LDARecommenderConfig conf;
    public LDATrainerFactory(LDARecommenderConfig conf){
        this.conf = conf;
    }

    /**
     * @return
     */
    public LDATrainer getTrainer(){
        if(conf.isLDAMaster()){
            return new LDATrainer(conf);
        }else{
            return new LDAImporter(conf);
        }
    }
}
