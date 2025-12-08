#if IOS || MACCATALYST
using CoreML;
using Foundation;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Platforms.iOS;

/// <summary>
/// iOS/macOS-specific ML inference using CoreML.
/// Provides native Apple Neural Engine acceleration for optimal performance.
/// </summary>
public class CoreMLInfer : IMLInfer, IDisposable
{
    private readonly object _lock = new();
    private bool _disposed;
    private MLModel? _model;
    private MLModelDescription? _modelDescription;
    private string? _loadedModelPath;

    /// <summary>
    /// Gets the backend type - CoreML
    /// </summary>
    public MLBackend Backend => MLBackend.CoreML;

    /// <summary>
    /// Gets whether a model is currently loaded
    /// </summary>
    public bool IsModelLoaded => _model != null;

    /// <summary>
    /// Initializes a new instance of the <see cref="CoreMLInfer"/> class
    /// </summary>
    public CoreMLInfer()
    {
    }

    /// <summary>
    /// Load a CoreML model from a file path
    /// </summary>
    /// <param name="modelPath">Path to the CoreML model (.mlmodel, .mlmodelc, or .mlpackage)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    /// <exception cref="ArgumentException">Thrown if the model path is null or empty</exception>
    /// <exception cref="FileNotFoundException">Thrown if the model file does not exist</exception>
    /// <exception cref="InvalidOperationException">Thrown if model loading fails</exception>
    public async Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

        if (!File.Exists(modelPath) && !Directory.Exists(modelPath))
            throw new FileNotFoundException($"Model file or directory not found: {modelPath}");

        await Task.Run(() =>
        {
            lock (_lock)
            {
                cancellationToken.ThrowIfCancellationRequested();

                UnloadModel();

                try
                {
                    var url = NSUrl.FromFilename(modelPath);
                    var configuration = new MLModelConfiguration
                    {
                        ComputeUnits = MLComputeUnits.All
                    };

                    _model = MLModel.Create(url, configuration, out var error);

                    if (error != null || _model == null)
                        throw new InvalidOperationException(
                            $"Failed to load CoreML model: {error?.LocalizedDescription ?? "Unknown error"}");

                    _modelDescription = _model.ModelDescription;
                    _loadedModelPath = modelPath;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to load CoreML model from '{modelPath}': {ex.Message}", ex);
                }
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Load a CoreML model from a stream
    /// </summary>
    /// <param name="modelStream">Stream containing the CoreML model</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    /// <exception cref="ArgumentNullException">Thrown if the model stream is null</exception>
    /// <exception cref="InvalidOperationException">Thrown if model loading fails</exception>
    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelStream);

        var tempPath = Path.Combine(Path.GetTempPath(), $"temp_model_{Guid.NewGuid()}.mlmodelc");
        try
        {
            lock (_lock)
            {
                cancellationToken.ThrowIfCancellationRequested();
                UnloadModel();
            }

            using (var fileStream = File.Create(tempPath))
            {
                await modelStream.CopyToAsync(fileStream, cancellationToken);
            }

            await LoadModelAsync(tempPath, cancellationToken);
            _loadedModelPath = tempPath;
        }
        catch
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
            throw;
        }
    }

    /// <summary>
    /// Load a CoreML model from MAUI assets
    /// </summary>
    /// <param name="assetName">Name of the asset file or package (.mlmodel, .mlmodelc, or .mlpackage)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    /// <exception cref="ArgumentException">Thrown if the asset name is null or empty</exception>
    /// <exception cref="FileNotFoundException">Thrown if the asset file does not exist</exception>
    public async Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));

        var directAssetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, assetName);
        if (Directory.Exists(directAssetPath))
        {
            await LoadModelAsync(directAssetPath, cancellationToken);
            return;
        }

        var resourceName = Path.GetFileNameWithoutExtension(assetName);
        var resourceExt = Path.GetExtension(assetName).TrimStart('.');
        
        string? bundlePath = null;
        if (!string.IsNullOrEmpty(resourceExt))
        {
            bundlePath = NSBundle.MainBundle.PathForResource(resourceName, resourceExt);
        }
        else
        {
            bundlePath = NSBundle.MainBundle.PathForResource(assetName, "mlpackage") ??
                        NSBundle.MainBundle.PathForResource(assetName, "mlmodelc") ??
                        NSBundle.MainBundle.PathForResource(assetName, "mlmodel");
        }

        if (!string.IsNullOrEmpty(bundlePath))
        {
            await LoadModelAsync(bundlePath, cancellationToken);
            return;
        }

        var assetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, assetName);
        if (!File.Exists(assetPath) && !Directory.Exists(assetPath))
            throw new FileNotFoundException($"Asset '{assetName}' not found in bundle or base directory");

        await LoadModelAsync(assetPath, cancellationToken);
    }

    /// <summary>
    /// Run inference using float inputs
    /// </summary>
    /// <param name="inputs">Dictionary of input tensors (name → tensor)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of output tensors (name → tensor)</returns>
    /// <exception cref="ArgumentException">Thrown if inputs are null or empty</exception>
    /// <exception cref="InvalidOperationException">Thrown if no model is loaded or inference fails</exception>
    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(
        Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

        if (_model == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        return Task.Run(() =>
        {
            lock (_lock)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    var inputFeatures = ConvertToMLFeatureProvider(inputs);
                    var prediction = _model!.GetPrediction(inputFeatures, out var error);

                    if (error != null || prediction == null)
                        throw new InvalidOperationException(
                            $"CoreML prediction failed: {error?.LocalizedDescription ?? "Unknown error"}");

                    return ConvertFromMLFeatureProvider(prediction);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Inference failed: {ex.Message}", ex);
                }
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Run inference using Int64 inputs (converts to Int32 for CoreML compatibility).
    /// Typically used for NLP models with token IDs and attention masks.
    /// </summary>
    /// <param name="inputs">Dictionary of input tensors (name → tensor). Values are converted from Int64 to Int32 for CoreML.</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of output tensors (name → tensor)</returns>
    /// <exception cref="ArgumentException">Thrown if inputs are null or empty</exception>
    /// <exception cref="InvalidOperationException">Thrown if no model is loaded or inference fails</exception>
    /// <remarks>
    /// CoreML NLP models expect Int32 token IDs, but .NET ML libraries typically use Int64.
    /// This method handles the conversion automatically.
    /// </remarks>
    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(
        Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

        if (_model == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        return Task.Run(() =>
        {
            lock (_lock)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    var inputFeatures = ConvertLongInputsToMLFeatureProvider(inputs);
                    var prediction = _model!.GetPrediction(inputFeatures, out var error);

                    if (error != null || prediction == null)
                        throw new InvalidOperationException(
                            $"CoreML prediction failed: {error?.LocalizedDescription ?? "Unknown error"}");

                    return ConvertFromMLFeatureProvider(prediction);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Inference failed: {ex.Message}", ex);
                }
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Get input metadata for the loaded model
    /// </summary>
    /// <returns>Dictionary of input names and their metadata</returns>
    /// <exception cref="InvalidOperationException">Thrown if no model is loaded</exception>
    public Dictionary<string, MLNodeMetadata> GetInputMetadata()
    {
        if (_model == null || _modelDescription == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        lock (_lock)
        {
            var metadata = new Dictionary<string, MLNodeMetadata>();

            foreach (var inputDesc in _modelDescription.InputDescriptionsByName)
            {
                var name = inputDesc.Key.ToString() ?? string.Empty;
                var feature = inputDesc.Value;

                if (feature is MLFeatureDescription mlFeature &&
                    mlFeature.Type == MLFeatureType.MultiArray &&
                    mlFeature.MultiArrayConstraint != null)
                {
                    var constraint = mlFeature.MultiArrayConstraint;
                    var shape = constraint.Shape.Select(s => (int)s).ToArray();
                    var elementType = ConvertMLDataType(constraint.DataType);

                    metadata[name] = new MLNodeMetadata(elementType, shape);
                }
            }

            return metadata;
        }
    }

    /// <summary>
    /// Get output metadata for the loaded model
    /// </summary>
    /// <returns>Dictionary of output names and their metadata</returns>
    /// <exception cref="InvalidOperationException">Thrown if no model is loaded</exception>
    public Dictionary<string, MLNodeMetadata> GetOutputMetadata()
    {
        if (_model == null || _modelDescription == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        lock (_lock)
        {
            var metadata = new Dictionary<string, MLNodeMetadata>();

            foreach (var outputDesc in _modelDescription.OutputDescriptionsByName)
            {
                var name = outputDesc.Key.ToString() ?? string.Empty;
                var feature = outputDesc.Value;

                if (feature is MLFeatureDescription mlFeature &&
                    mlFeature.Type == MLFeatureType.MultiArray &&
                    mlFeature.MultiArrayConstraint != null)
                {
                    var constraint = mlFeature.MultiArrayConstraint;
                    var shape = constraint.Shape.Select(s => (int)s).ToArray();
                    var elementType = ConvertMLDataType(constraint.DataType);

                    metadata[name] = new MLNodeMetadata(elementType, shape);
                }
            }

            return metadata;
        }
    }

    /// <summary>
    /// Unload the currently loaded model and release resources
    /// </summary>
    public void UnloadModel()
    {
        lock (_lock)
        {
            _model?.Dispose();
            _model = null;
            _modelDescription = null;

            if (!string.IsNullOrEmpty(_loadedModelPath) &&
                _loadedModelPath.Contains(Path.GetTempPath()))
            {
                try
                {
                    if (File.Exists(_loadedModelPath))
                        File.Delete(_loadedModelPath);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }

            _loadedModelPath = null;
        }
    }

    /// <summary>
    /// Dispose of resources
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Protected implementation of Dispose pattern
    /// </summary>
    /// <param name="disposing">True if called from Dispose; false if called from finalizer</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            UnloadModel();
        }

        _disposed = true;
    }

    /// <summary>
    /// Finalizer to ensure resources are released
    /// </summary>
    ~CoreMLInfer()
    {
        Dispose(false);
    }

    #region Private Helper Methods

    /// <summary>
    /// Convert long tensor inputs to CoreML MLFeatureProvider using Int32.
    /// CoreML NLP models expect Int32 for token IDs, not Int64.
    /// </summary>
    /// <remarks>
    /// This conversion is necessary because:
    /// - .NET ML libraries (ONNX Runtime, ML.NET) use Int64 for token IDs
    /// - CoreML models are compiled with Int32 input types for token IDs
    /// </remarks>
    private IMLFeatureProvider ConvertLongInputsToMLFeatureProvider(Dictionary<string, Tensor<long>> inputs)
    {
        var features = new NSMutableDictionary<NSString, NSObject>();

        foreach (var input in inputs)
        {
            var name = (NSString)input.Key;
            var tensor = input.Value;
            var shape = tensor.Dimensions.ToArray();
            var shapeNS = new NSNumber[shape.Length];
            
            for (int i = 0; i < shape.Length; i++)
            {
                shapeNS[i] = NSNumber.FromInt32(shape[i]);
            }
            
            var array = tensor.ToArray();

            var mlArray = new MLMultiArray(shapeNS, MLMultiArrayDataType.Int32, out var error);
            if (error != null || mlArray == null)
                throw new InvalidOperationException(
                    $"Failed to create MLMultiArray for input '{input.Key}': {error?.LocalizedDescription ?? "Unknown error"}");

            for (int i = 0; i < array.Length; i++)
            {
                mlArray[i] = NSNumber.FromInt32((int)array[i]);
            }

            var featureValue = MLFeatureValue.Create(mlArray);
            features[name] = featureValue;
        }

        return new MLDictionaryFeatureProvider(
            new NSDictionary<NSString, NSObject>(features.Keys, features.Values), out _);
    }

    /// <summary>
    /// Convert float tensor inputs to CoreML MLFeatureProvider
    /// </summary>
    private IMLFeatureProvider ConvertToMLFeatureProvider(Dictionary<string, Tensor<float>> inputs)
    {
        var features = new NSMutableDictionary<NSString, NSObject>();

        foreach (var input in inputs)
        {
            var name = (NSString)input.Key;
            var tensor = input.Value;
            var shape = tensor.Dimensions.ToArray();
            var shapeNS = new NSNumber[shape.Length];
            
            for (int i = 0; i < shape.Length; i++)
            {
                shapeNS[i] = NSNumber.FromInt32(shape[i]);
            }
            
            var array = tensor.ToArray();

            var mlArray = new MLMultiArray(shapeNS, MLMultiArrayDataType.Float32, out var error);
            if (error != null || mlArray == null)
                throw new InvalidOperationException(
                    $"Failed to create MLMultiArray for input '{input.Key}': {error?.LocalizedDescription ?? "Unknown error"}");

            for (int i = 0; i < array.Length; i++)
            {
                mlArray[i] = NSNumber.FromFloat(array[i]);
            }

            var featureValue = MLFeatureValue.Create(mlArray);
            features[name] = featureValue;
        }

        return new MLDictionaryFeatureProvider(
            new NSDictionary<NSString, NSObject>(features.Keys, features.Values), out _);
    }

    /// <summary>
    /// Convert CoreML MLFeatureProvider outputs to tensor dictionary
    /// </summary>
    private Dictionary<string, Tensor<float>> ConvertFromMLFeatureProvider(IMLFeatureProvider provider)
    {
        var outputs = new Dictionary<string, Tensor<float>>();

        if (_modelDescription == null)
            throw new InvalidOperationException("Model description is not available");

        foreach (var outputDesc in _modelDescription.OutputDescriptionsByName)
        {
            var name = outputDesc.Key.ToString();
            if (string.IsNullOrEmpty(name))
                continue;

            var featureValue = provider.GetFeatureValue(name);
            if (featureValue?.MultiArrayValue == null)
                continue;

            var mlArray = featureValue.MultiArrayValue;
            var shape = mlArray.Shape.Select(s => (int)s).ToArray();
            var count = shape.Aggregate(1, (a, b) => a * b);

            var data = new float[count];
            for (int i = 0; i < count; i++)
            {
                data[i] = mlArray[i].FloatValue;
            }

            var tensor = new DenseTensor<float>(data, shape);
            outputs[name] = tensor;
        }

        return outputs;
    }

    /// <summary>
    /// Convert CoreML data type to .NET System.Type
    /// </summary>
    private static Type ConvertMLDataType(MLMultiArrayDataType dataType)
    {
        return dataType switch
        {
            MLMultiArrayDataType.Double => typeof(double),
            MLMultiArrayDataType.Float32 => typeof(float),
            MLMultiArrayDataType.Int32 => typeof(int),
            _ => typeof(float)
        };
    }

    #endregion
}
#endif
