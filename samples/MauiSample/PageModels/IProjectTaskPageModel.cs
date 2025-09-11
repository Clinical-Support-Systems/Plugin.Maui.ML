using CommunityToolkit.Mvvm.Input;
using MauiSample.Models;

namespace MauiSample.PageModels
{
    public interface IProjectTaskPageModel
    {
        IAsyncRelayCommand<ProjectTask> NavigateToTaskCommand { get; }
        bool IsBusy { get; }
    }
}