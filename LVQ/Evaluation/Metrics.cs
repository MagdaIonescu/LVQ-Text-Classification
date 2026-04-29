using LVQ.Algorithms;
using LVQ.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace LVQ.Evaluation
{
    internal class Metrics
    {
        public static void Evaluate(LVQClassifier model, List<DocumentVector> testData, int prototypes)
        {
            var actualLabels = testData.Select(x => x.Label).Distinct();
            var predictedLabels = testData.Select(x => model.Predict(x.Features)).Distinct();
            var labels = actualLabels.Union(predictedLabels).Distinct().ToList();

            Dictionary<string, Dictionary<string, int>> confusion = new Dictionary<string, Dictionary<string, int>>();

            foreach (var actual in labels)
            {
                confusion[actual] = new Dictionary<string, int>();
                foreach (var predicted in labels)
                {
                    confusion[actual][predicted] = 0;
                }
            }

            foreach (var doc in testData)
            {
                string actual = doc.Label;
                string predicted = model.Predict(doc.Features);

                confusion[actual][predicted]++;
            }

            string txt = "";
            txt += "Confusion Matrix:\n\n";
            int colWidth = 10;
            txt += "Actual".PadRight(colWidth);

            foreach (var label in labels)
            {
                txt += label.PadRight(colWidth);
            }
            txt += "\n";

            foreach (var actual in labels)
            {
                txt += actual.PadRight(colWidth);
                foreach (var predicted in labels)
                {
                    txt += confusion[actual][predicted].ToString().PadRight(colWidth);
                }
                txt += "\n";
            }
            txt += "\n";

            int correct = 0;
            int total = testData.Count;

            foreach (var label in labels)
            {
                correct += confusion[label][label];
            }

            double accuracy = (double)correct / total;
            txt += $"Accuracy: {(accuracy * 100):F2}%\n\n";

            foreach (var label in labels)
            {
                int TP = confusion[label][label];
                int FP = labels.Where(l => l != label).Sum(l => confusion[l][label]);
                int FN = labels.Where(l => l != label).Sum(l => confusion[label][l]);

                double precision = TP == 0 ? 0 : (double)TP / (TP + FP);
                double recall = TP == 0 ? 0 : (double)TP / (TP + FN);
                double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

                txt += $"Class: {label}\n";
                txt += $"  Precision: {(precision * 100):F2}%\n";
                txt += $"  Recall:    {(recall * 100):F2}%\n";
                txt += $"  F1-score:  {(f1 * 100):F2}%\n\n";
            }

            string folder = @"..\..\Results";
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
            string path = Path.Combine(folder, $"results_{prototypes}.txt");

            txt = $"Prototypes per class: {prototypes}\n\n" + txt;
            File.WriteAllText(path, txt);

            Console.WriteLine("Evaluation complete.");
        }
    }
}
