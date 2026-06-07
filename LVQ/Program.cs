using System;
using System.Collections.Generic;
using LVQ.Algorithms;
using LVQ.Data;
using LVQ.Evaluation;
using LVQ.Models;

namespace LVQ
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Loading data...");

            ArffReader reader = new ArffReader();

            List<DocumentVector> trainingData = reader.ReadDocuments("Datasets/MultiClass_Training_SVM_1309.0.arff");
            List<DocumentVector> testData = reader.ReadDocuments("Datasets/MultiClass_Testing_SVM_1309.0.arff");

            Console.WriteLine($"Training samples: {trainingData.Count}");
            Console.WriteLine($"Testing samples: {testData.Count}");
            Console.WriteLine();

            int[] prototypeValues = {1, 3, 5};

            foreach (var p in prototypeValues)
            {
                Console.WriteLine($"Prototypes per class: {p}");

                LVQClassifier model = new LVQClassifier(p, learningRate: 0.1, epochs: 20, windowWidth: 0.3, epsilon: 0.1);

                // Train
                model.Initialize(trainingData);

                Console.WriteLine("Training model with LVQ 1...");
                model.TrainLVQ1(trainingData);
                Console.WriteLine("Training model with LVQ 2 optimization...");
                model.TrainLVQ2(trainingData);
                Console.WriteLine("Training model with LVQ 2.1 optimization...");
                model.TrainLVQ2_1(trainingData);
                Console.WriteLine("Training model with LVQ 3 optimization...");
                model.TrainLVQ3(trainingData);


                Console.WriteLine("Training complete.");
                Console.WriteLine();

                // Evaluate
                Console.WriteLine("Evaluating model...");
                Metrics.Evaluate(model, testData, p);

                Console.WriteLine("Done.");
                Console.WriteLine();
            }
        }
    }
}