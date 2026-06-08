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
        private double windowWidth; 
        private double epsilon; 

        public LVQClassifier(int prototypesPerClass, double learningRate = 0.1, int epochs = 20, double windowWidth = 0.3, double epsilon = 0.2)
        {
            this.prototypesPerClass = prototypesPerClass;
            this.learningRate = learningRate;
            this.epochs = epochs;
            this.windowWidth = windowWidth;
            this.epsilon = epsilon;
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
                    }
                    else
                    {
                        MoveAway(winner, input, learningRate);
                    }
                }

                learningRate *= 0.95;
            }
        }

        public void TrainLVQ2(List<DocumentVector> trainingData)
        {
            double lvq2LearningRate = 0.03;
            int lvq2Epochs = 5;

            for (int epoch = 0; epoch < lvq2Epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);
                    GetTwoClosestPrototypes(input, out Prototype winner, out double winnerDistance, out Prototype runnerUp, out double runnerUpDistance);
                    if (!IsInsideWindow(winnerDistance, runnerUpDistance))
                        continue;

                    if (winner.Label != runnerUp.Label && winner.Label != doc.Label && runnerUp.Label == doc.Label)
                    {
                        MoveAway(winner, input, lvq2LearningRate);
                        MoveCloser(runnerUp, input, lvq2LearningRate);
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
                    GetTwoClosestPrototypes(input, out Prototype winner, out double winnerDistance, out Prototype runnerUp, out double runnerUpDistance);

                    if (!IsInsideWindow(winnerDistance, runnerUpDistance))
                        continue;

                    if (winner.Label == runnerUp.Label)
                        continue;

                    bool winnerCorrect = winner.Label == doc.Label;
                    bool runnerUpCorrect = runnerUp.Label == doc.Label;

                    if (winnerCorrect == runnerUpCorrect)
                        continue;

                    if (winnerCorrect)
                    {
                        MoveCloser(winner, input, lvq21LearningRate);
                        MoveAway(runnerUp, input, lvq21LearningRate);
                    }
                    else
                    {
                        MoveCloser(runnerUp, input, lvq21LearningRate);
                        MoveAway(winner, input, lvq21LearningRate);
                    }
                }
                lvq21LearningRate *= 0.95;
            }
        }

        public void TrainLVQ3(List<DocumentVector> trainingData)
        {
            double lvq3LearningRate = 0.03;
            int lvq3Epochs = 5;

            for (int epoch = 0; epoch < lvq3Epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);
                    GetTwoClosestPrototypes(input, out Prototype winner, out double winnerDistance, out Prototype runnerUp, out double runnerUpDistance);

                    if (!IsInsideWindow(winnerDistance, runnerUpDistance))
                        continue;

                    bool winnerCorrect = winner.Label == doc.Label;
                    bool runnerUpCorrect = runnerUp.Label == doc.Label;

                    if (winnerCorrect && runnerUpCorrect)
                    {
                        double adjustedLearningRate = lvq3LearningRate * epsilon;
                        MoveCloser(winner, input, adjustedLearningRate);
                        MoveCloser(runnerUp, input, adjustedLearningRate);
                    }
                    else if (winnerCorrect != runnerUpCorrect)
                    {
                        if (winnerCorrect)
                        {
                            MoveCloser(winner, input, lvq3LearningRate);
                            MoveAway(runnerUp, input, lvq3LearningRate);
                        }
                        else
                        {
                            MoveCloser(runnerUp, input, lvq3LearningRate);
                            MoveAway(winner, input, lvq3LearningRate);
                        }
                    }
                }
                lvq3LearningRate *= 0.95;
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

        private void GetTwoClosestPrototypes(double[] input, out Prototype winner, out double winnerDistance, out Prototype runnerUp, out double runnerUpDistance)
        {
            winner = null;
            runnerUp = null;
            winnerDistance = double.MaxValue;
            runnerUpDistance = double.MaxValue;

            foreach (var prototype in Prototypes)
            {
                double distance = ComputeDistance(input, prototype.Features);
                if (distance < winnerDistance)
                {
                    runnerUp = winner;
                    runnerUpDistance = winnerDistance;
                    winner = prototype;
                    winnerDistance = distance;
                }
                else if (distance < runnerUpDistance)
                {
                    runnerUp = prototype;
                    runnerUpDistance = distance;
                }
            }
        }

        private bool IsInsideWindow(double winnerDistance, double runnerUpDistance)
        {
            if (winnerDistance == 0 && runnerUpDistance == 0)
                return true;
            if (winnerDistance == 0 || runnerUpDistance == 0)
                return false;

            double s = (1.0 - windowWidth) / (1.0 + windowWidth);
            double ratio = Math.Min(winnerDistance / runnerUpDistance, runnerUpDistance / winnerDistance);

            return ratio > s;
        }
    }
}
