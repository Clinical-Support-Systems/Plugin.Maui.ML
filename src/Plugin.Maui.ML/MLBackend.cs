namespace Plugin.Maui.ML;

/// <summary>
///     ML inference backend types
/// </summary>
public enum MLBackend
{
    /// <summary>
    ///     Provides functionality for loading and executing ONNX models using the ONNX Runtime library.
    /// </summary>
    /// <remarks>
    ///     Use this class to perform inference with ONNX models in .NET applications. It manages model
    ///     sessions, input and output bindings, and execution of inference requests. Thread safety and resource management
    ///     depend on the specific usage and configuration; refer to the documentation of individual members for
    ///     details.
    /// </remarks>
    OnnxRuntime,

    /// <summary>
    ///     Provides functionality for working with CoreML models, enabling integration of machine learning workflows within
    ///     .NET applications.
    /// </summary>
    /// <remarks>
    ///     Use this class to load, evaluate, and manage CoreML models. It is designed to facilitate
    ///     interoperability between .NET and CoreML, allowing developers to leverage trained models for prediction and
    ///     inference tasks. Thread safety and performance considerations may vary depending on the specific methods used;
    ///     refer to individual member documentation for details.
    /// </remarks>
    CoreML,

    /// <summary>
    ///     Provides access to machine learning functionalities and utilities within the application.
    /// </summary>
    /// <remarks>
    ///     Use this class to interact with various machine learning features, such as model loading,
    ///     inference, and data preprocessing. The available methods and properties depend on the specific implementation
    ///     and supported ML tasks.
    /// </remarks>
    MLKit,

    /// <summary>
    ///     Provides access to Windows Machine Learning functionality for loading, evaluating, and managing machine learning
    ///     models within Windows applications.
    /// </summary>
    /// <remarks>
    ///     Use this class to interact with Windows ML features, such as model inference and resource
    ///     management. It serves as the entry point for integrating machine learning workflows into Windows-based
    ///     solutions.
    /// </remarks>
    WindowsML
}