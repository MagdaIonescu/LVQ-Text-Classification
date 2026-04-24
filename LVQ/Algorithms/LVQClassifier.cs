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

        public LVQClassifier(double learningRate = 0.1, int epochs = 20)
        {
            this.learningRate = learningRate;
            this.epochs = epochs;
        }

        public void Initialize(List<DocumentVector> trainingData)
        {
            var grouped = trainingData.GroupBy(x => x.Label);

            foreach (var group in grouped)
            {
                DocumentVector first = group.First();

                Prototypes.Add(new Prototype((double[])first.Features.Clone(), first.Label));
            }
        }

        public void Train(List<DocumentVector> trainingData)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var doc in trainingData)
                {
                    Prototype winner = GetClosestPrototype(doc.Features);

                    if (winner.Label == doc.Label)
                    {
                        MoveCloser(winner, doc.Features);
                    } else
                    {
                        MoveAway(winner, doc.Features);
                    }
                }

                learningRate *= 0.95; 
            }
        }

        public string Predict(double[] input)
        {
            Prototype winner = GetClosestPrototype(input);
            return winner.Label;
        }

        private double ComputeDistance(double[] a, double[] b)
        {
            double sum = 0;

            for (int i = 0; i < a.Length; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }

            return Math.Sqrt(sum);
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
