using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML;

/// <summary>
///     Interface for ML inference operations.
///     Supports multiple backends: ONNX Runtime, CoreML, ML Kit, Windows ML.
/// </summary>
public interface IMLInfer
{
    /// <summary>
    ///     The backend implementation being used
    /// </summary>
    MLBackend Backend { get; }

    /// <summary>
    ///     Check if a model is currently loaded
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    ///     Load a model from a file path
    /// </summary>
    /// <param name="modelPath">Path to the model file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default);

    /// <summary>
    ///     Load a model from a stream
    /// </summary>
    /// <param name="modelStream">Stream containing the model</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default);

    /// <summary>
    ///     Load a model from MAUI assets
    /// </summary>
    /// <param name="assetName">Name of the asset file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default);

    /// <summary>
    ///     Run inference using float inputs (all inputs must be float-compatible and
    ///     provided as Tensor&lt;float&gt;).
    /// </summary>
    /// <param name="inputs">Input tensors for the model</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of output tensors</returns>
    Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(
        Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default);

    /// <summary>
    ///     Run inference using Int64 inputs (e.g. token ids / attention masks).
    /// </summary>
    /// <param name="inputs">Input tensors for the model</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of output tensors</returns>
    Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(
        Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default);

    /// <summary>
    ///     Get input metadata for the loaded model
    /// </summary>
    /// <returns>Dictionary of input names and their metadata</returns>
    Dictionary<string, MLNodeMetadata> GetInputMetadata();

    /// <summary>
    ///     Get output metadata for the loaded model
    /// </summary>
    /// <returns>Dictionary of output names and their metadata</returns>
    Dictionary<string, MLNodeMetadata> GetOutputMetadata();

    /// <summary>
    ///     Dispose of the loaded model and release resources
    /// </summary>
    void UnloadModel();
}
