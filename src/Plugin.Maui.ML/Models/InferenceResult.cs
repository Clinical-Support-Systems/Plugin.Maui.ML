using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Models;

/// <summary>
///     Represents the result of an ML inference operation
/// </summary>
public class InferenceResult
{
    /// <summary>
    ///     Gets or sets the output tensors from the inference
    /// </summary>
    public Dictionary<string, Tensor<float>> Outputs { get; set; } = new();

    /// <summary>
    ///     Gets or sets the time taken for the inference operation
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    ///     Gets or sets any additional metadata about the inference
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    ///     Gets or sets whether the inference was successful
    /// </summary>
    public bool IsSuccess { get; set; } = true;

    /// <summary>
    ///     Gets or sets any error message if the inference failed
    /// </summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
///     Represents model information and metadata
/// </summary>
public class ModelInfo
{
    /// <summary>
    ///     Gets or sets the model name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    ///     Gets or sets the model version
    /// </summary>
    public string Version { get; set; } = string.Empty;

    /// <summary>
    ///     Gets or sets the model description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    ///     Gets or sets the input specifications
    /// </summary>
    public List<TensorSpec> Inputs { get; set; } = new();

    /// <summary>
    ///     Gets or sets the output specifications
    /// </summary>
    public List<TensorSpec> Outputs { get; set; } = new();

    /// <summary>
    ///     Gets or sets additional model metadata
    /// </summary>
    public Dictionary<string, string> CustomMetadata { get; set; } = new();
}

/// <summary>
///     Represents tensor specification information
/// </summary>
public class TensorSpec
{
    /// <summary>
    ///     Gets or sets the tensor name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    ///     Gets or sets the tensor shape
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    ///     Gets or sets the tensor data type
    /// </summary>
    public string DataType { get; set; } = string.Empty;

    /// <summary>
    ///     Gets or sets the tensor description
    /// </summary>
    public string Description { get; set; } = string.Empty;
}
