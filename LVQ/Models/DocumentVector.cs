namespace LVQ.Models
{
    public class DocumentVector
    {
        public double[] Features { get; set; } 
        public string Label { get; set; } 
        public DocumentVector(double[] features, string label)
        {
            Features = features;
            Label = label;
        }
    }
}