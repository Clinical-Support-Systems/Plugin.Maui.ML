using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MauiSample.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Plugin.Maui.ML;
// For NodeMetadata

namespace MauiSample.PageModels
{
    public partial class MainPageModel : ObservableObject, IProjectTaskPageModel
    {
        private const int DefaultSequenceLength = 128;
        private readonly CategoryRepository _categoryRepository;
        private readonly ModalErrorHandler _errorHandler;
        private readonly IMLInfer _mlInfer;
        private readonly ProjectRepository _projectRepository;
        private readonly SeedDataService _seedDataService;
        private readonly TaskRepository _taskRepository;
        [ObservableProperty] private string _answer = string.Empty;
        private bool _dataLoaded;
        [ObservableProperty] private bool _isBusy;
        [ObservableProperty] private bool _isModelLoading;
        private bool _isNavigatedTo;
        [ObservableProperty] private bool _isRefreshing;
        [ObservableProperty] private List<Project> _projects = new();
        [ObservableProperty] private string _question = string.Empty;
        [ObservableProperty] private List<ProjectTask> _tasks = new();
        [ObservableProperty] private string _today = DateTime.Now.ToString("dddd, MMM d");
        [ObservableProperty] private List<Brush> _todoCategoryColors = new();

        [ObservableProperty] private List<CategoryChartData> _todoCategoryData = new();

        public MainPageModel(SeedDataService seedDataService, ProjectRepository projectRepository,
            TaskRepository taskRepository, CategoryRepository categoryRepository, ModalErrorHandler errorHandler,
            IMLInfer mlInfer)
        {
            _projectRepository = projectRepository;
            _taskRepository = taskRepository;
            _categoryRepository = categoryRepository;
            _errorHandler = errorHandler;
            _seedDataService = seedDataService;
            _mlInfer = mlInfer;
        }

        public bool HasCompletedTasks => Tasks?.Any(t => t.IsCompleted) ?? false;

        private async Task EnsureModelLoaded()
        {
            if (_mlInfer.IsModelLoaded || IsModelLoading) return;
            try
            {
                IsModelLoading = true;
                Answer = "Loading model...";
                await _mlInfer.LoadModelFromAssetAsync("t5encoder_Opset17.onnx");
                Answer = "Model loaded. Ask.";
            }
            catch (Exception ex) { Answer = $"Model load error: {ex.Message}"; }
            finally { IsModelLoading = false; }
        }

        private async Task LoadData()
        {
            try
            {
                IsBusy = true;
                Projects = await _projectRepository.ListAsync();
                var chartData = new List<CategoryChartData>();
                var chartColors = new List<Brush>();
                var cats = await _categoryRepository.ListAsync();
                foreach (var c in cats)
                {
                    chartColors.Add(c.ColorBrush);
                    var tasksCount = Projects.Where(p => p.CategoryID == c.ID).SelectMany(p => p.Tasks).Count();
                    chartData.Add(new CategoryChartData(c.Title, tasksCount));
                }

                TodoCategoryData = chartData;
                TodoCategoryColors = chartColors;
                Tasks = await _taskRepository.ListAsync();
            }
            finally
            {
                IsBusy = false;
                OnPropertyChanged(nameof(HasCompletedTasks));
            }
        }

        private async Task InitData(SeedDataService seed)
        {
            if (!Preferences.Default.ContainsKey("is_seeded"))
                await seed.LoadSeedDataAsync();
            Preferences.Default.Set("is_seeded", true);
            await Refresh();
        }

        [RelayCommand]
        private async Task Refresh()
        {
            try
            {
                IsRefreshing = true;
                await LoadData();
            }
            catch (Exception e) { _errorHandler.HandleError(e); }
            finally { IsRefreshing = false; }
        }

        [RelayCommand]
        private void NavigatedTo()
        {
            _isNavigatedTo = true;
        }

        [RelayCommand]
        private void NavigatedFrom()
        {
            _isNavigatedTo = false;
        }

        [RelayCommand]
        private async Task Appearing()
        {
            if (!_dataLoaded)
            {
                await InitData(_seedDataService);
                _dataLoaded = true;
                await Refresh();
            }
            else if (!_isNavigatedTo)
            {
                await Refresh();
            }

            await EnsureModelLoaded();
        }

        [RelayCommand]
        private Task TaskCompleted(ProjectTask task)
        {
            OnPropertyChanged(nameof(HasCompletedTasks));
            return _taskRepository.SaveItemAsync(task);
        }

        [RelayCommand]
        private Task AddTask()
        {
            return Shell.Current.GoToAsync("task");
        }

        [RelayCommand]
        private Task NavigateToProject(Project project)
        {
            return Shell.Current.GoToAsync($"project?id={project.ID}");
        }

        [RelayCommand]
        private Task NavigateToTask(ProjectTask task)
        {
            return Shell.Current.GoToAsync($"task?id={task.ID}");
        }

        [RelayCommand]
        private async Task CleanTasks()
        {
            var completed = Tasks.Where(t => t.IsCompleted).ToList();
            foreach (var t in completed)
            {
                await _taskRepository.DeleteItemAsync(t);
                Tasks.Remove(t);
            }

            OnPropertyChanged(nameof(HasCompletedTasks));
            Tasks = new List<ProjectTask>(Tasks);
            await AppShell.DisplayToastAsync("All cleaned up!");
        }

        private static long[] Tokenize(string text, int maxLen)
        {
            var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var ids = new long[maxLen];
            for (var i = 0; i < Math.Min(tokens.Length, maxLen); i++)
                ids[i] = Math.Abs(tokens[i].GetHashCode()) % 32000 + 1; // pseudo vocab id
            return ids;
        }

        [RelayCommand(CanExecute = nameof(CanAskQuestion))]
        private async Task AskQuestion()
        {
            if (string.IsNullOrWhiteSpace(Question))
            {
                Answer = "Enter question.";
                return;
            }

            await EnsureModelLoaded();
            if (!_mlInfer.IsModelLoaded)
            {
                Answer = "Model not loaded.";
                return;
            }

            try
            {
                IsBusy = true;
                Answer = "Running...";

                var meta = _mlInfer.GetInputMetadata();
                if (meta.Count == 0)
                {
                    Answer = "Model has no inputs.";
                    return;
                }

                // Determine primary input (prefer input_ids)
                NodeMetadata firstMeta;
                if (!meta.TryGetValue("input_ids", out firstMeta))
                    firstMeta = meta.First().Value;

                // Determine sequence length
                var dims = firstMeta.Dimensions;
                var seqLen = DefaultSequenceLength;
                if (dims.Length > 1 && dims[1] > 0) seqLen = dims[1];

                var tokenIds = Tokenize(Question, seqLen);

                var allLong = meta.Values.All(m => m.ElementType == typeof(long) || m.ElementType == typeof(long));
                var allFloat = meta.Values.All(m => m.ElementType == typeof(float) || m.ElementType == typeof(double));

                if (allLong)
                {
                    var inputs = new Dictionary<string, Tensor<long>>();
                    foreach (var kv in meta)
                    {
                        var isMask = kv.Key.Contains("mask", StringComparison.OrdinalIgnoreCase) ||
                                     kv.Key.Contains("attention", StringComparison.OrdinalIgnoreCase);
                        if (isMask)
                        {
                            var mask = new long[seqLen];
                            var filled = tokenIds.TakeWhile(v => v != 0).Count();
                            for (var i = 0; i < filled; i++) mask[i] = 1;
                            inputs[kv.Key] = new DenseTensor<long>(mask, new[] { 1, seqLen });
                        }
                        else
                        {
                            inputs[kv.Key] = new DenseTensor<long>(tokenIds, new[] { 1, seqLen });
                        }
                    }

                    var outputs = await _mlInfer.RunInferenceLongInputsAsync(inputs);
                    DisplayFirstOutput(outputs.First());
                }
                else // fallback: build float inputs
                {
                    var inputs = new Dictionary<string, Tensor<float>>();
                    foreach (var kv in meta)
                    {
                        var isMask = kv.Key.Contains("mask", StringComparison.OrdinalIgnoreCase) ||
                                     kv.Key.Contains("attention", StringComparison.OrdinalIgnoreCase);
                        if (isMask)
                        {
                            var mask = new float[seqLen];
                            var filled = tokenIds.TakeWhile(v => v != 0).Count();
                            for (var i = 0; i < filled; i++) mask[i] = 1f;
                            inputs[kv.Key] = new DenseTensor<float>(mask, new[] { 1, seqLen });
                        }
                        else
                        {
                            var floats = tokenIds.Select(id => (float)id).ToArray();
                            inputs[kv.Key] = new DenseTensor<float>(floats, new[] { 1, seqLen });
                        }
                    }

                    var outputs = await _mlInfer.RunInferenceAsync(inputs);
                    DisplayFirstOutput(outputs.First());
                }
            }
            catch (Exception ex)
            {
                Answer = $"Inference error: {ex.Message}";
            }
            finally { IsBusy = false; }
        }

        private void DisplayFirstOutput(KeyValuePair<string, Tensor<float>> kv)
        {
            var t = kv.Value;
            var preview = string.Join(", ", t.ToArray().Take(10).Select(v => v.ToString("0.####")));
            var shape = string.Join(",", t.Dimensions.ToArray());
            Answer = $"Out '{kv.Key}' [{shape}] => {preview}";
        }

        private bool CanAskQuestion()
        {
            return !IsBusy && !IsModelLoading;
        }

        partial void OnIsBusyChanged(bool value)
        {
            AskQuestionCommand.NotifyCanExecuteChanged();
        }

        partial void OnIsModelLoadingChanged(bool value)
        {
            AskQuestionCommand.NotifyCanExecuteChanged();
        }
    }
}
