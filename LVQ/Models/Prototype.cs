namespace LVQ.Models
{
    class Prototype
    {
        public double[] Features { get; set; }
        public string Label { get; set; }

        public Prototype(double[] features, string label)
        {
            Features = features;
            Label = label;
        }
    }
}
