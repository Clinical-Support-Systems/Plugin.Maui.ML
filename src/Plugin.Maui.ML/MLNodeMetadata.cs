namespace Plugin.Maui.ML;

/// <summary>
/// Platform-agnostic metadata for ML model nodes
/// Used by backends that cannot access ONNX Runtime's NodeMetadata
/// </summary>
public class MLNodeMetadata
{
    /// <summary>
    /// Gets the .NET type of the tensor element
    /// </summary>
    public Type ElementType { get; }

    /// <summary>
    /// Gets the dimensions/shape of the tensor
    /// </summary>
    public int[] Dimensions { get; }

    /// <summary>
    /// Gets the symbolic dimension names, if available
    /// </summary>
    public string[]? SymbolicDimensions { get; }

    /// <summary>
    /// Initializes a new instance of MLNodeMetadata
    /// </summary>
    public MLNodeMetadata(Type elementType, int[] dimensions, string[]? symbolicDimensions = null)
    {
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
        Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
        SymbolicDimensions = symbolicDimensions;
    }
}