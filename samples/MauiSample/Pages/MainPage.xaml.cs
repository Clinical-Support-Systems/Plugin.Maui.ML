using MauiSample.Models;
using MauiSample.PageModels;

namespace MauiSample.Pages
{
    public partial class MainPage : ContentPage
    {
        public MainPage(MainPageModel model)
        {
            InitializeComponent();
            BindingContext = model;
        }
    }
}