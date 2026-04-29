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

        public LVQClassifier(int prototypesPerClass, double learningRate = 0.1, int epochs = 20)
        {
            this.prototypesPerClass = prototypesPerClass;
            this.learningRate = learningRate;
            this.epochs = epochs;
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

        public void Train(List<DocumentVector> trainingData)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    var input = Normalize(doc.Features);
                    Prototype winner = GetClosestPrototype(input);

                    if (winner.Label == doc.Label)
                    {
                        MoveCloser(winner, input);
                    } else
                    {
                        MoveAway(winner, input);
                    }
                }

                learningRate *= 0.95; 
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

        private void MoveCloser(Prototype prototype, double[] input)
        {
            for (int i = 0; i < prototype.Features.Length; i++)
            {
                prototype.Features[i] += learningRate * (input[i] - prototype.Features[i]);
            }
        }

        private void MoveAway(Prototype prototype, double[] input)
        {
            for (int i = 0; i < prototype.Features.Length; i++)
            {
                prototype.Features[i] -= learningRate * (input[i] - prototype.Features[i]);
            }
        }
    }
}
