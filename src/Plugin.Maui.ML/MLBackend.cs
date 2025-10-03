namespace Plugin.Maui.ML;

/// <summary>
///     ML inference backend types
/// </summary>
public enum MLBackend
{
    /// <summary>ONNX Runtime (cross-platform)</summary>
    OnnxRuntime,
    
    /// <summary>Apple CoreML (iOS/macOS)</summary>
    CoreML,
    
    /// <summary>Google ML Kit / TensorFlow Lite (Android)</summary>
    MLKit,
    
    /// <summary>Windows ML (Windows)</summary>
    WindowsML
}
