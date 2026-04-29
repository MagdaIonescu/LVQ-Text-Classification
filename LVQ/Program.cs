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

                LVQClassifier model = new LVQClassifier(p, learningRate: 0.1, epochs: 20);

                // Train
                Console.WriteLine("Training model...");
                model.Initialize(trainingData);
                model.Train(trainingData);

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