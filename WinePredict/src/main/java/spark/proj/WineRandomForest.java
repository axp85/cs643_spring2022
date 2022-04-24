import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

/**
 * @author Amit Pandejee
 * @course CS643-852
 * @Date: 2022-04-21
 * @Project:  Project 2
 * @Purpose: Random Forest Predictor with F1 score
 *
 */

public class WineRandomForest {

	public static List<String> outputlines;
	public static RandomForestModel rfModel;
	
	public static List<String> RandomForestPredictorF1(JavaSparkContext jsc, JavaRDD<LabeledPoint> valData,
			String moelFileName) {
		
		outputlines = new ArrayList();
		
		rfModel = RandomForestModel.load(jsc.sc(),  moelFileName);
		
		 outputlines.add("Random Forest Prediction for Wine Quality");
		 outputlines.add("\nPredicted : Expected");
		 
		 // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
        		valData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        System.out.println(rfModel.predict(p.features())+" : "+p.label());
                    	//outputlines.add(rfmodel.predict(p.features())+" : "+p.label());
                        return new Tuple2<>(rfModel.predict(p.features()), p.label());
                    }
                });
        
        predictionAndLabel.foreach(data -> {
        	outputlines.add(data._1() + " : " + data._2());
        });
        
        //F1 Scoring Section
        F1Scoring(predictionAndLabel);
      
        
        return outputlines;
	}
	
	public static  void  F1Scoring( JavaPairRDD<Double, Double> predictionAndLabels ) {
		
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		
		//Starting Matrix for RF
		System.out.println("********Random Forest Matrix***************\n");
		outputlines.add("********Random Forest Matrix***************\n");

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		//System.out.println("Confusion matrix: \n" + confusion);
		//outputlines.add("Confusion matrix: \n" + confusion);

		// Overall statistics
		//System.out.println("Accuracy = " + metrics.accuracy());
		//outputlines.add("Accuracy = " + metrics.accuracy());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
		 /*
		  System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
		    metrics.labels()[i]));
		  outputlines.add(String.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
				    metrics.labels()[i])));
		  System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
		    metrics.labels()[i]));
		  outputlines.add(String.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
				    metrics.labels()[i])));
		  */
		  System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
		    metrics.labels()[i]));
		  outputlines.add(String.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
				    metrics.labels()[i])));
		}

		//Weighted stats
		/*
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		outputlines.add(String.format("Weighted precision = %f\n", metrics.weightedPrecision()));
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		outputlines.add(String.format("Weighted recall = %f\n", metrics.weightedRecall()));
		*/
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		outputlines.add(String.format("Weighted F1 score = %f\n", metrics.weightedFMeasure()));
		/*
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		outputlines.add(String.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate()));
		*/
	}

	
	private static PairFunction<LabeledPoint, Double, Double> pf =  new PairFunction<LabeledPoint, Double, Double>() {
	        @Override
	        public Tuple2<Double, Double> call(LabeledPoint p) {
	            Double prediction= null;
	            try {
	                prediction = rfModel.predict(p.features());
	            } catch (Exception e) {
	                //logger.error(ExceptionUtils.getStackTrace(e));
	                e.printStackTrace();
	            }
	            System.out.println(prediction+" : "+p.label());
	            return new Tuple2<>(prediction, p.label());
	        }
	    };
	
	 private static Function<Tuple2<Double, Double>, Boolean> f = new Function<Tuple2<Double, Double>, Boolean>() {
	     @Override
	     public Boolean call(Tuple2<Double, Double> pl) {
	         return !pl._1().equals(pl._2());
	     }
	 };
	 
	public static void wirteOutputToFile(String fileName, List<String> output) {
		
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
		
	 
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String validationFileName = args[0];
		//String validationFileName = "cs643_data/ValidationDataset.csv";
		
		WinePredictML  wp = new WinePredictML();
		
		WineRandomForest wrf = new WineRandomForest();
		JavaSparkContext jsc = wp.createJavaSparkContext("WinePredictionML");
		JavaRDD<LabeledPoint> validationdData=	wp.createRDD2(jsc, validationFileName);
		
		//Model location
		String dirLocation = "target/model/";
		String wineRandomForestModelFileNameF1 = dirLocation+"RandomForestModel";
		
		//Predicted output and F1 scoring location
		String dirLocationPred = "target/";
		
		String wineRandomForestPredictionFileName = dirLocationPred + "RandomForestPredictionResult.txt";
		
		List<String> outputValidationLinesRF = wrf.RandomForestPredictorF1(jsc, validationdData, wineRandomForestModelFileNameF1);
		wrf.wirteOutputToFile(wineRandomForestPredictionFileName, outputValidationLinesRF);
		
		wp.javaSparkStop(jsc);
		
	}
	
	
	

}
