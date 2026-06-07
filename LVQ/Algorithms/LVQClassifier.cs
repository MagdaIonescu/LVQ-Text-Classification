using LVQ.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LVQ.Algorithms
{
    class LVQClassifier
    {
        public List<Prototype> Prototypes { get; set; } = new List<Prototype>();

        private double learningRate;
        private int epochs;
        private int prototypesPerClass;
        private double windowWidth; // Used in LVQ2 and LVQ2.1 to determine if the second closest prototype is close enough to the input for an update

        public LVQClassifier(int prototypesPerClass, double learningRate = 0.1, int epochs = 20, double windowWidth = 0.3)
        {
            this.prototypesPerClass = prototypesPerClass;
            this.learningRate = learningRate;
            this.epochs = epochs;
            this.windowWidth = windowWidth;
        }

        public void Initialize(List<DocumentVector> trainingData)
        {
            var grouped = trainingData.GroupBy(x => x.Label);
            Random rnd = new Random(0);

            foreach (var group in grouped)
            {
                var samples = group.OrderBy(x => rnd.Next()).Take(prototypesPerClass);
                foreach (var sample in samples)
                {
                    var prototypeVector = Normalize(sample.Features);
                    Prototypes.Add(new Prototype(prototypeVector, sample.Label));
                }
            }
        }

        public void TrainLVQ1(List<DocumentVector> trainingData)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);
                    Prototype winner = GetClosestPrototype(input);

                    if (winner.Label == doc.Label)
                    {
                        MoveCloser(winner, input, learningRate);
                    } else
                    {
                        MoveAway(winner, input, learningRate);
                    }
                }

                learningRate *= 0.95; 
            }
        }

        public void TrainLVQ2(List<DocumentVector> trainingData)
        {
            // fine-tuning for LVQ2, smaller learning rate and fewer epochs than LVQ1
            double lvq2LearningRate = 0.03;
            int lvq2Epochs = 5;

            for (int epoch = 0; epoch < lvq2Epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);

                    GetTwoClosestPrototypes(input, out Prototype m1, out double d1, out Prototype m2, out double d2);

                    if (!IsInsideWindow(d1, d2))
                        continue;

                    if (m1.Label != m2.Label && m1.Label != doc.Label && m2.Label == doc.Label)
                    {
                        MoveAway(m1, input, lvq2LearningRate);  
                        MoveCloser(m2, input, lvq2LearningRate);
                    }
                }

                lvq2LearningRate *= 0.95;
            }
        }

        public void TrainLVQ2_1(List<DocumentVector> trainingData)
        {
            double lvq21LearningRate = 0.03;
            int lvq21Epochs = 5;

            for (int epoch = 0; epoch < lvq21Epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);

                    GetTwoClosestPrototypes(input, out Prototype m1, out double d1, out Prototype m2, out double d2);

                    if (!IsInsideWindow(d1, d2))
                        continue;

                    if (m1.Label == m2.Label)
                        continue;

                    bool m1Correct = m1.Label == doc.Label;
                    bool m2Correct = m2.Label == doc.Label;

                    if (m1Correct == m2Correct)
                        continue;

                    if (m1Correct)
                    {
                        MoveCloser(m1, input, lvq21LearningRate);
                        MoveAway(m2, input, lvq21LearningRate);
                    } else
                    {
                        MoveCloser(m2, input, lvq21LearningRate);
                        MoveAway(m1, input, lvq21LearningRate);
                    }
                }

                lvq21LearningRate *= 0.95;
            }
        }

        public string Predict(double[] input)
        {
            var inputNorm = Normalize(input);
            Prototype winner = GetClosestPrototype(inputNorm);
            return winner.Label;
        }

        private double ComputeDistance(double[] a, double[] b)
        {
            double dot = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            if (normA == 0 || normB == 0)
                return double.MaxValue;

            return 1 - (dot / (Math.Sqrt(normA) * Math.Sqrt(normB)));
        }

        private double[] Normalize(double[] v)
        {
            double sum = 0;

            for (int i = 0; i < v.Length; i++)
            {
                sum += v[i] * v[i];
            }

            double norm = Math.Sqrt(sum);
            if (norm == 0)
                return v;

            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] / norm;
            }

            return result;
        }

        private Prototype GetClosestPrototype(double[] input)
        {
            Prototype best = Prototypes[0];
            double minDist = ComputeDistance(input, best.Features);

            foreach (var prototype in Prototypes)
            {
                double dist = ComputeDistance(input, prototype.Features);
                if (dist < minDist)
                {
                    minDist = dist;
                    best = prototype;
                }
            }

            return best;
        }

        private void MoveCloser(Prototype prototype, double[] input, double currentLearningRate)
        {
            for (int i = 0; i < prototype.Features.Length; i++)
            {
                prototype.Features[i] += currentLearningRate * (input[i] - prototype.Features[i]);
            }
        }

        private void MoveAway(Prototype prototype, double[] input, double currentLearningRate)
        {
            for (int i = 0; i < prototype.Features.Length; i++)
            {
                prototype.Features[i] -= currentLearningRate * (input[i] - prototype.Features[i]);
            }
        }

        private void GetTwoClosestPrototypes(double[] input, out Prototype best1, out double min1, out Prototype best2, out double min2)
        {
            best1 = null;
            best2 = null;
            min1 = double.MaxValue;
            min2 = double.MaxValue;

            foreach (var prototype in Prototypes)
            {
                double distrance = ComputeDistance(input, prototype.Features);
                if (distrance < min1)
                {
                    best2 = best1;
                    min2 = min1;
                    best1 = prototype;
                    min1 = distrance;
                } 
                else if (distrance < min2)
                {
                    best2 = prototype;
                    min2 = distrance;
                }
            }
        }

        private bool IsInsideWindow(double d1, double d2)
        {
            if (d1 == 0 && d2 == 0) 
                return true;
            if (d1 == 0 || d2 == 0) 
                return false;

            double s = (1.0 - windowWidth) / (1.0 + windowWidth);
            double ratio = Math.Min(d1 / d2, d2 / d1);

            return ratio > s;
        }
    }
}
