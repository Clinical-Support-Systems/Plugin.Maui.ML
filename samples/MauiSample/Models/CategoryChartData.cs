namespace MauiSample.Models
{
    public class CategoryChartData
    {
        public CategoryChartData(string title, int count)
        {
            Title = title;
            Count = count;
        }

        public string Title { get; set; } = string.Empty;
        public int Count { get; set; }
    }
}
