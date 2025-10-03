using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MauiSample.Models;
using Microsoft.ML.OnnxRuntime.Tensors;
using Plugin.Maui.ML;

// ReSharper disable UnusedParameterInPartialMethod

// For NodeMetadata

namespace MauiSample.PageModels
{
    public partial class MainPageModel : ObservableObject, IProjectTaskPageModel
    {
        private const int DefaultSequenceLength = 128;
        private readonly ModalErrorHandler _errorHandler;
        private readonly IMedicalNlpService _medicalNlpService;
        private readonly IMLInfer _mlInfer;

        [ObservableProperty] private string _answer = string.Empty;
        private bool _dataLoaded;
        [ObservableProperty] private List<MedicalEntity> _extractedEntities = [];
        [ObservableProperty] private bool _isBusy;
        [ObservableProperty] private bool _isMedicalNlpLoading; // NEW: track medical NLP model loading state
        [ObservableProperty] private bool _isMedicalNlpReady;
        [ObservableProperty] private bool _isModelLoading;
        private bool _isNavigatedTo;
        [ObservableProperty] private bool _isRefreshing;
        [ObservableProperty] private string _medicalStatus = "Not initialized";

        // Medical NLP properties
        [ObservableProperty] private string _medicalText = "";
        [ObservableProperty] private string _question = string.Empty;
        [ObservableProperty] private string _today = DateTime.Now.ToString("dddd, MMM d");

        public MainPageModel(ModalErrorHandler errorHandler,
            IMLInfer mlInfer, IMedicalNlpService medicalNlpService)
        {
            _errorHandler = errorHandler;
            _mlInfer = mlInfer;
            _medicalNlpService = medicalNlpService;
        }

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
            catch (Exception ex)
            {
                Answer = $"Model load error: {ex.Message}";
            }
            finally
            {
                IsModelLoading = false;
            }
        }

        private async Task EnsureMedicalNlpInitialized()
        {
            if (_medicalNlpService.IsInitialized || IsMedicalNlpLoading) return;

            try
            {
                IsMedicalNlpLoading = true;
                MedicalStatus = "Initializing medical NLP (this may take a moment)...";
                await _medicalNlpService.InitializeAsync();
                IsMedicalNlpReady = true;
                MedicalStatus = "Medical NLP ready";
            }
            catch (Exception ex)
            {
                IsMedicalNlpReady = false;
                MedicalStatus = $"Failed: {ex.Message}";
            }
            finally
            {
                IsMedicalNlpLoading = false;
            }
        }

        [RelayCommand]
        private Task Refresh()
        {
            try
            {
                IsRefreshing = true;
            }
            catch (Exception e)
            {
                _errorHandler.HandleError(e);
            }
            finally
            {
                IsRefreshing = false;
            }

            return Task.CompletedTask;
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
                
                _dataLoaded = true;
                await Refresh();
            }
            else if (!_isNavigatedTo)
            {
                await Refresh();
            }

            await EnsureModelLoaded();

            // Initialize medical NLP in background (will set IsMedicalNlpLoading to show busy indicator)
            _ = EnsureMedicalNlpInitialized();
        }


        // Medical NLP Commands

        [RelayCommand(CanExecute = nameof(CanExtractEntities))]
        private async Task ExtractMedicalEntities()
        {
            if (string.IsNullOrWhiteSpace(MedicalText))
            {
                MedicalStatus = "Enter medical text";
                return;
            }

            await EnsureMedicalNlpInitialized();

            if (!IsMedicalNlpReady)
            {
                MedicalStatus = "Medical NLP not ready";
                return;
            }

            try
            {
                IsBusy = true;
                MedicalStatus = "Extracting entities...";

                var entities = await _medicalNlpService.ExtractEntitiesAsync(MedicalText);
                ExtractedEntities = entities;

                MedicalStatus = $"Found {entities.Count} entities";
            }
            catch (Exception ex)
            {
                MedicalStatus = $"Error: {ex.Message}";
                ExtractedEntities = [];
            }
            finally
            {
                IsBusy = false;
            }
        }

        [RelayCommand]
        private async Task TestMedicalSample()
        {
            MedicalText =
                "62-year-old male presented to ER with severe chest pain radiating to left arm. Diagnosed with acute myocardial infarction. Administered aspirin 325mg, morphine 5mg IV, and initiated heparin infusion. Emergency cardiac catheterization performed.";
            await ExtractMedicalEntities();
        }

        private bool CanExtractEntities()
        {
            // Block while general busy, medical NLP loading, or not ready yet
            return !IsBusy && !IsMedicalNlpLoading && IsMedicalNlpReady;
        }

        // Original T5 inference

        private static long[] Tokenize(string text, int maxLen)
        {
            var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var ids = new long[maxLen];
            for (var i = 0; i < Math.Min(tokens.Length, maxLen); i++)
                ids[i] = Math.Abs(tokens[i].GetHashCode()) % 32000 + 1;
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

                if (!meta.TryGetValue("input_ids", out var firstMeta))
                    firstMeta = meta.First().Value;

                var dims = firstMeta.Dimensions;
                var seqLen = DefaultSequenceLength;
                if (dims.Length > 1 && dims[1] > 0) seqLen = dims[1];

                var tokenIds = Tokenize(Question, seqLen);

                var allLong = meta.Values.All(m => m.ElementType == typeof(long) || m.ElementType == typeof(long));

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
                else
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
            finally
            {
                IsBusy = false;
            }
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
            ExtractMedicalEntitiesCommand.NotifyCanExecuteChanged();
        }

        partial void OnIsModelLoadingChanged(bool value)
        {
            AskQuestionCommand.NotifyCanExecuteChanged();
        }

        partial void OnIsMedicalNlpReadyChanged(bool value)
        {
            ExtractMedicalEntitiesCommand.NotifyCanExecuteChanged();
        }

        partial void OnIsMedicalNlpLoadingChanged(bool value)
        {
            // Update button state while loading NLP model
            ExtractMedicalEntitiesCommand.NotifyCanExecuteChanged();
        }
    }
}