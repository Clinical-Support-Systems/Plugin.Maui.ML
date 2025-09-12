#nullable disable
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MauiSample.Models;

namespace MauiSample.PageModels
{
    public partial class ProjectListPageModel : ObservableObject
    {
        private readonly ProjectRepository _projectRepository;

        [ObservableProperty] private List<Project> _projects = [];

        public ProjectListPageModel(ProjectRepository projectRepository)
        {
            _projectRepository = projectRepository;
        }

        [RelayCommand]
        private async Task Appearing()
        {
            Projects = await _projectRepository.ListAsync();
        }

        [RelayCommand]
        private Task NavigateToProject(Project project)
        {
            return Shell.Current.GoToAsync($"project?id={project.ID}");
        }

        [RelayCommand]
        private async Task AddProject()
        {
            await Shell.Current.GoToAsync("project");
        }
    }
}
