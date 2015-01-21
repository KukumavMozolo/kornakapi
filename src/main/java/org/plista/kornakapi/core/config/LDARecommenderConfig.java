package org.plista.kornakapi.core.config;

public class LDARecommenderConfig extends RecommenderConfig{
	
	private String trainingset;
	private String luceneIndexPath;
	private Integer numberOfTopics;
	private String ldaModelDirectory;
	private double alpha;
	private double etha;
	private int trainingThreats;
	private int inferenceThreats;
	private double maxTermVariance;
	private String preprocessingDataDirectory;  
	private Integer minimumWords;
    private Integer maxIterations;
    private String yarnInputDir;
    private String yarnOutputDir;
    private String hadoopUser;
    private String hadoopConfPath;
    private boolean LDAMaster;

    public String getMaxIterations(){
        return Integer.toString(maxIterations);
    }

	public String getLuceneIndexPath(){
		return luceneIndexPath;
	}

	public Integer getMinimumWords() {
		return minimumWords;
	}
	
	public String getPreprocessingDataDirectory() {
		return preprocessingDataDirectory;
	}
	  
	public String getTrainingSetName() {
		return trainingset;
	}
	
	public String getVectorOutputPath() {
		return ldaModelDirectory + "/seq/";
	}
	public String getTextDirectoryPath() {
		return ldaModelDirectory + "/Documents/";
	}
	
	public String getSparseVectorOutputPath() {
		return ldaModelDirectory + "/sparse/";
	}

	public String getTopicsOutputPath() {
		return ldaModelDirectory + "/topics/";
	}

	public Integer getnumberOfTopics() {
		return numberOfTopics;
	}
	
	public String getCVBInputPath(){
		return ldaModelDirectory + "/sparse/tf-vectors-cvb/";
	}
	
	public String getTopicsDictionaryPath(){
		return ldaModelDirectory + "/sparse/dictionary.file-0";
	}
	
	public String getTmpLDAModelPath(){
		return ldaModelDirectory + "/tmpModel/";
	}
	
	public String getLDADocTopicsPath(){
		return ldaModelDirectory + "DocumentTopics";
	}
	public String getLDARecommenderModelPath(){
		return ldaModelDirectory + "/Model/";
	}
	public String getInferencePath(){
		return ldaModelDirectory + "/Inference/";
	}
	public String getNewDocPath(){
		return getTextDirectoryPath();
	}
	public double getAlpha(){
		return alpha;
	}
	public double getEta(){
		return etha;
	}
	public Integer getTrainingThreats(){
		return trainingThreats;
	}
	public int getInferenceThreats(){
		return inferenceThreats;
	}
	public double getMaxDFSigma(){
		return maxTermVariance;
	}

    public String getYarnInputDir() {
        return yarnInputDir;
    }
    public String getYarnOutputDir() {
        return yarnOutputDir;
    }
    public String getHadoopUser(){ return hadoopUser; }

    public String getLdaPrintPath(){
        return  ldaModelDirectory + "print/";
    }
    public String getHadoopConfPath(){ return hadoopConfPath;}

    public boolean isLDAMaster() {
        return LDAMaster;
    }
}
