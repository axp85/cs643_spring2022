import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;

import scala.Tuple2;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.attribute.AttributeGroup;
import org.apache.spark.ml.attribute.NumericAttribute;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;

/**
 * @author Amit Pandejee
 * @course CS643-852
 * @Date: 2022-04-21
 * @Project:  Project 2
 * @Purpose: Training Models
 *
 */
public class WinePredictML {
	
	public static RandomForestModel rfModel;
	public static DecisionTreeModel dtModel;
	public static LogisticRegressionModel lrModel;
	public static List<String> outputlines;
	public JavaSparkContext createJavaSparkContext (String contextName) {
		
		// $example on$
		//The master URL to connect to, such as "local" to run locally with one thread, "local[4]" to run locally with 4 cores, 
		//or "spark://master:7077" to run on a Spark standalone cluster.
	    //SparkConf sparkConf = new SparkConf().setAppName(contextName)
	    //        .setMaster("local[*]").set("spark.executor.memory","2g");
		 SparkConf sparkConf = new SparkConf().setAppName(contextName);
		            //.setMaster("local[*]");
	    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

	    
	    return jsc;
	}
	

	
	public SparkSession createSparkSessionMasterCtx(SparkContext sc) {
		
		 if (sc == null)
			 return null;
		 
		 SparkSession spark = SparkSession.builder().config(sc.getConf()).getOrCreate();
		 
		 return spark;
	}

	public List<String> buildLogisticalRegressionF1(JavaSparkContext jsc, JavaRDD<LabeledPoint> trainDataSet, JavaRDD<LabeledPoint> validationDataSet,
							String moelFileName) {
		
		List<String> outputToFile= new ArrayList<String>();
		
		JavaRDD<LabeledPoint> training = trainDataSet;
		JavaRDD<LabeledPoint> test = validationDataSet;
		
		// Run training algorithm to build the model.
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
		  .setNumClasses(10)
		  .run(training.rdd());
		
		
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
		  new Tuple2<>(model.predict(p.features()), p.label()));
		
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		
		//Starting Matrix for LR
		System.out.println("********Logistical Regression Matrix***************\n");
		outputToFile.add("********Logistical Regression Matrix***************\n");

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		//System.out.println("Confusion matrix: \n" + confusion);
		//outputToFile.add("Confusion matrix: \n" + confusion);

		// Overall statistics
		//System.out.println("Accuracy = " + metrics.accuracy());
		//outputToFile.add("Accuracy = " + metrics.accuracy());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
		  /*
		  System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
		  		    metrics.labels()[i])));
		  System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
		  		    metrics.labels()[i])));
		  */
		  System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
				    metrics.labels()[i])));
		}

		//Weighted stats
		/*
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		outputToFile.add(String.format("Weighted precision = %f\n", metrics.weightedPrecision()));
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		outputToFile.add(String.format("Weighted recall = %f\n", metrics.weightedRecall()));
		*/
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		outputToFile.add(String.format("Weighted F1 score = %f\n", metrics.weightedFMeasure()));
		/*
		 System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		 outputToFile.add(String.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate()));
		*/
		
		
	    // Delete if model already present, and Save the new model
	    try {
	        FileUtils.forceDelete(new File(moelFileName));
	        System.out.println("\nDeleting old model completed.");
	    } catch (FileNotFoundException e1) {
	    } catch (IOException e) {
	    }

		// Save and load model
		model.save(jsc.sc(), moelFileName);
		
		return outputToFile;
	}
	
	
	
	public List<String> buildDecisionTreeF1(JavaSparkContext jsc, JavaRDD<LabeledPoint> trainDataSet, JavaRDD<LabeledPoint> validationDataSet,
			String moelFileName) {

		List<String> outputToFile= new ArrayList<String>();

		
		JavaRDD<LabeledPoint> training = trainDataSet;
		JavaRDD<LabeledPoint> test = validationDataSet;
		
		// Set parameters.
		//  Empty categoricalFeaturesInfo indicates all features are continuous.
		int numClasses = 10;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "gini";
		int maxDepth = 5;
		int maxBins = 32;
		
		// Train a DecisionTree model for classification.
		DecisionTreeModel model = DecisionTree.trainClassifier(training, numClasses,
		categoricalFeaturesInfo, impurity, maxDepth, maxBins);
		

		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
		new Tuple2<>(model.predict(p.features()), p.label()));
				
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		
		//Starting Matrix for DT
		System.out.println("********Decision Tree Matrix***************\n");
		outputToFile.add("********Decision Tree Matrix***************\n");

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		//System.out.println("Confusion matrix: \n" + confusion);
		//outputToFile.add("Confusion matrix: \n" + confusion);

		// Overall statistics
		//System.out.println("Accuracy = " + metrics.accuracy());
		//outputToFile.add("Accuracy = " + metrics.accuracy());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
		 /* System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
				    metrics.labels()[i])));
		  System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
				    metrics.labels()[i])));
		  */
		  System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
				    metrics.labels()[i])));
		}

		//Weighted stats
		/*
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		outputToFile.add(String.format("Weighted precision = %f\n", metrics.weightedPrecision()));
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		outputToFile.add(String.format("Weighted recall = %f\n", metrics.weightedRecall()));
		*/
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		outputToFile.add(String.format("Weighted F1 score = %f\n", metrics.weightedFMeasure()));
		/*
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		outputToFile.add(String.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate()));
		*/
		
		
	    // Delete if model already present, and Save the new model
	    try {
	        FileUtils.forceDelete(new File(moelFileName));
	        System.out.println("\nDeleting old model completed.");
	    } catch (FileNotFoundException e1) {
	    } catch (IOException e) {
	    }
		
		// Save and load model
		model.save(jsc.sc(), moelFileName);
				
		return outputToFile;
	}
	
	
	public List<String>  buildRandomForestF1(JavaSparkContext jsc, JavaRDD<LabeledPoint> trainDataSet, JavaRDD<LabeledPoint> validationDataSet,
			String moelFileName) {

		List<String> outputToFile= new ArrayList<String>();
		
		JavaRDD<LabeledPoint> training = trainDataSet;
		JavaRDD<LabeledPoint> test = validationDataSet;
		
		// Train a RandomForest model.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		int numClasses = 10;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		int numTrees = 3; // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		int maxDepth = 5;
		int maxBins = 32;
		int seed = 12345;
		
		RandomForestModel model = RandomForest.trainClassifier(training, numClasses,
		categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
		seed);
		
		
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
		new Tuple2<>(model.predict(p.features()), p.label()));
				
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		
		//Starting Matrix for RF
		System.out.println("********Random Forest Matrix***************\n");
		outputToFile.add("********Random Forest Matrix***************\n");

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		//System.out.println("Confusion matrix: \n" + confusion);
		//outputToFile.add("Confusion matrix: \n" + confusion);

		// Overall statistics
		//System.out.println("Accuracy = " + metrics.accuracy());
		//outputToFile.add("Accuracy = " + metrics.accuracy());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
		 /*
		  System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
				    metrics.labels()[i])));
		  System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
				    metrics.labels()[i])));
		  */
		  System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
		    metrics.labels()[i]));
		  outputToFile.add(String.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
				    metrics.labels()[i])));
		}

		//Weighted stats
		/*
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		outputToFile.add(String.format("Weighted precision = %f\n", metrics.weightedPrecision()));
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		outputToFile.add(String.format("Weighted recall = %f\n", metrics.weightedRecall()));
		*/
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		outputToFile.add(String.format("Weighted F1 score = %f\n", metrics.weightedFMeasure()));
		/*
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		outputToFile.add(String.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate()));
		*/
		
		
	    // Delete if model already present, and Save the new model
	    try {
	        FileUtils.forceDelete(new File(moelFileName));
	        System.out.println("\nDeleting old model completed.");
	    } catch (FileNotFoundException e1) {
	    } catch (IOException e) {
	    }
		
		// Save and load model
		model.save(jsc.sc(), moelFileName);
				
		return outputToFile;

	  }

	public void sparkStop(SparkSession spark) {
		
		if(spark != null)
			spark.stop();
		
	}
	
	public void javaSparkStop(JavaSparkContext spark) {
		
		if(spark != null)
			spark.stop();
		
	}
	
	public JavaRDD<String> createRDD(JavaSparkContext jsc, String dataFile) {
		
		  //JavaRDD data = MLUtils.loadLibSVMFile(jsc.sc(), dataFile).toJavaRDD();
		  JavaRDD<String> data = jsc.textFile(dataFile);
		 
		  
		  return data;
	}
	
	public static JavaRDD<LabeledPoint> createRDD2(JavaSparkContext jsc, String dataFile) {
		
		
				
		//JavaSparkContext sc = new JavaSparkContext(sparkConf);
        //String path = "com.databricks.spark.csv";
        JavaRDD<String> data = jsc.textFile(dataFile);
        JavaRDD<String> data2 = data.filter(new Function<String, Boolean>(){public Boolean call(String s) {return !s.matches(".*[a-zA-Z]+.*");}});
        JavaRDD<String> data3 = data2.map(new Function<String, String>() {
        	public String call(String line) throws Exception {
        		  String[] parts = line.split(";");
        		  for(int i=0; i<parts.length; i++) {
        			  if(!parts[i].contains("."))
        				  parts[i] = parts[i] + ".0";
        		  }
        		  
        		  return String.join(";", parts);
        	}
        });
        JavaRDD<LabeledPoint> parsedData = data3
                .map(new Function<String, LabeledPoint>()  {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(";");
                        System.out.println(line);
                        return new LabeledPoint(Double.parseDouble(parts[11]),
                        		Vectors.dense(Double.parseDouble(parts[0]),
                                		Double.parseDouble(parts[1]),
                                		Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4]),
                                        Double.parseDouble(parts[5]),
                                        Double.parseDouble(parts[6]),
                                        Double.parseDouble(parts[7]),
                                        Double.parseDouble(parts[8]),
                                        Double.parseDouble(parts[9]),
                                        Double.parseDouble(parts[10])));
                    }
                });

        return  parsedData;
	}
	
	public void wirteOutputToFile(String fileName, List<String> output) {
		
		  try {
		      FileWriter fileWriter = new FileWriter(fileName+".txt");
		      PrintWriter printWriter = new PrintWriter(fileWriter);
		      for(String s : output) {
		    	  printWriter.println(s);
		      }
		      printWriter.close();
		    } catch (IOException e) {
		      System.out.println("An error occurred.");
		      e.printStackTrace();
		    }
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		//if(args.length < 1)
		//	System.out.print("Please provide input traing data file");
		
		//if (args.length > 1)
		//	System.out.print("Application only accept traing data file");
		
		//String inputFileName = args[0];
		//String validationFileName = args[1];
		
		String inputFileName = "cs643_data/TrainingDataset.csv";
		String validationFileName = "cs643_data/ValidationDataset.csv";
		 
		WinePredictML  wp = new WinePredictML();

		JavaSparkContext jsc = wp.createJavaSparkContext("WinePredictionML");
			
		JavaRDD<LabeledPoint> trainData=	wp.createRDD2(jsc, inputFileName);
		
		JavaRDD<LabeledPoint> validationdData=	wp.createRDD2(jsc, validationFileName);
		
	
		String dirLocation = "target/model/";
		
		//Build Logistic Regression Model
		String wineLogisticRegressionModelFileNameF1 = dirLocation+"LogistricRegressionModel";
		List<String> logisticalRegressionOutputF1  = wp.buildLogisticalRegressionF1(jsc, trainData, validationdData, wineLogisticRegressionModelFileNameF1);
		
		//Build Decision Tree Model
		String wineDecisionTreeModelFileNameF1 = dirLocation+"DecisionTreeModel";
		List<String> decisionTreeOutputF1  = wp.buildDecisionTreeF1(jsc, trainData, validationdData,  wineDecisionTreeModelFileNameF1);
		
		//Build Random Forest Model
		String wineRandomForestModelFileNameF1 = dirLocation+"RandomForestModel";
		List<String> randomForestOutputF1  = wp.buildRandomForestF1(jsc, trainData, validationdData,  wineRandomForestModelFileNameF1);
		
		List<String> finalOutput = new ArrayList<String>();
		
		finalOutput.addAll(logisticalRegressionOutputF1);
		finalOutput.addAll(decisionTreeOutputF1);
		finalOutput.addAll(randomForestOutputF1);
		
		String FinalOutputF1 = dirLocation + "F1State.txt";
		wp.wirteOutputToFile(FinalOutputF1, finalOutput);
		
		
		wp.javaSparkStop(jsc);

	}

}
