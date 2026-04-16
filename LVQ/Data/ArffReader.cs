using System.Collections.Generic;
using System.IO;
using LVQ.Models;

namespace LVQ.Data
{
    public class ArffReader
    {
        public int GetFeatureCount(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                if (line.StartsWith("#Attributes"))
                {
                    string[] parts = line.Split(' ');
                    return int.Parse(parts[1]);
                }
            }
            return 0;
        }
        public List<DocumentVector> ReadDocuments(string filePath)
        {
            List<DocumentVector> documents = new List<DocumentVector>();
            string[] lines = File.ReadAllLines(filePath);
            int featureCount = GetFeatureCount(filePath);
            bool dataSection = false;

            foreach (string line in lines)
            {
                if (!dataSection)
                {
                    if (line == "@data")
                        dataSection = true;
                    continue;
                }

                string[] parts = line.Split('#');
                if (parts.Length < 2)
                    continue;

                string attributesPart = parts[0].Trim();
                string labelsPart = parts[1].Trim();
                
                string[] labels = labelsPart.Split(' ');
                string label = labels[0];
                
                double[] features = new double[featureCount];
                
                string[] pairs = attributesPart.Split(' ');
                foreach (string pair in pairs)
                {
                    string[] indexValue = pair.Split(':');
                    int index = int.Parse(indexValue[0]);
                    double value = double.Parse(indexValue[1]);
                    features[index] = value;
                }
                documents.Add(new DocumentVector(features, label));
            }
            return documents;
        }
    }
}