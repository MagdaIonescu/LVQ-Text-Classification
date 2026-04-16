using System;
using LVQ.Data;

namespace LVQ
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // test
            ArffReader reader = new ArffReader();
            string path = @"Datasets\MultiClass_Training_SVM_100.0.arff";

            var docs = reader.ReadDocuments(path);

            Console.WriteLine("Documente citite: " + docs.Count);
            if (docs.Count > 0)
            {
                Console.WriteLine("Prima eticheta: " + docs[0].Label);
                Console.WriteLine("Prima valoare: " + docs[0].Features[0]);
            }
        }
    }
}