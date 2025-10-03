# Platform-Specific ML Backends

Plugin.Maui.ML now supports multiple ML inference backends, allowing you to choose between cross-platform ONNX Runtime and platform-native solutions for optimal performance.

## Available Backends

### 1. ONNX Runtime (Default)
- **Platforms**: All (iOS, Android, Windows, macOS)
- **Best for**: Cross-platform models, ONNX format models
- **Hardware Acceleration**: 
  - iOS/macOS: CoreML execution provider
  - Android: NNAPI execution provider
  - Windows: DirectML execution provider

### 2. CoreML (iOS/macOS)
- **Platforms**: iOS 11+, macOS 10.13+
- **Best for**: Native CoreML models (.mlmodel, .mlmodelc)
- **Hardware Acceleration**: Apple Neural Engine (A12+, M1+)

### 3. ML Kit / TensorFlow Lite (Android) - Coming Soon
- **Platforms**: Android 8.1+ (API 27+)
- **Best for**: TensorFlow Lite models
- **Hardware Acceleration**: NNAPI, GPU delegates

### 4. Windows ML (Windows) - Coming Soon
- **Platforms**: Windows 10 1809+, Windows 11
- **Best for**: Windows-optimized models
- **Hardware Acceleration**: DirectML, DirectX 12

## Usage Examples

### Default (Platform-Optimized)

```csharp
// In MauiProgram.cs
builder.Services.AddMauiML();

// In your code
public class MyViewModel
{
    private readonly IMLInfer _mlService;
    
    public MyViewModel(IMLInfer mlService)
    {
        _mlService = mlService;
        // Automatically uses:
        // - ONNX + CoreML on iOS/macOS
        // - ONNX + NNAPI on Android
        // - ONNX + DirectML on Windows
    }
}
```

### Explicit Backend Selection

```csharp
// Force ONNX Runtime on all platforms
builder.Services.AddMauiML(MLBackend.OnnxRuntime);

// Use pure CoreML on iOS/macOS (requires .mlmodel files)
#if IOS || MACCATALYST
builder.Services.AddMauiML(MLBackend.CoreML);
#endif
```

### Configuration-Based Selection

```csharp
builder.Services.AddMauiML(config =>
{
    config.PreferredBackend = MLBackend.CoreML; // Platform default if not available
    config.EnablePerformanceLogging = true;
    config.MaxConcurrentInferences = 2;
});
```

### Runtime Backend Switching

```csharp
// Create different backends at runtime
IMLInfer onnxInfer = new OnnxRuntimeInfer();

#if IOS || MACCATALYST
IMLInfer coreMLInfer = new CoreMLInfer();
#endif

// Or use platform factory methods
#if IOS
IMLInfer nativeInfer = Platforms.iOS.PlatformMLInfer.CreateCoreMLInfer();
#endif
```

## Model Format Requirements

| Backend | Supported Formats | How to Convert |
|---------|------------------|----------------|
| ONNX Runtime | `.onnx` | Use `torch.onnx.export()`, `tf2onnx`, etc. |
| CoreML | `.mlmodel`, `.mlmodelc` | Use `coremltools`, `onnx-coreml` |
| ML Kit | `.tflite` | Use TensorFlow Lite converter |
| Windows ML | `.onnx`, `.onnxmodel` | Same as ONNX Runtime |

## Converting Models

### PyTorch to ONNX
```python
import torch
model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### ONNX to CoreML
```python
import coremltools as ct
from onnx_coreml import convert

model = convert(model='model.onnx')
model.save('model.mlmodel')
```

### TensorFlow to TFLite
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Platform Capabilities

### Check Available Features

```csharp
#if IOS
// Check if Neural Engine is available
var hasNeuralEngine = Platforms.iOS.PlatformMLInfer.IsNeuralEngineAvailable();

// Get available execution providers
var providers = Platforms.iOS.PlatformMLInfer.GetAvailableExecutionProviders();
#endif

#if ANDROID
// Check if NNAPI is available
var hasNnapi = Platforms.Android.PlatformMLInfer.IsNnapiAvailable();
#endif

#if WINDOWS
// Check if DirectX 12 is available
var hasDX12 = Platforms.Windows.PlatformMLInfer.IsDirectX12Available();

// Get system info
var sysInfo = Platforms.Windows.PlatformMLInfer.GetSystemInfo();
#endif

// Check current backend
if (_mlService.Backend == MLBackend.CoreML)
{
    Console.WriteLine("Using Apple's Neural Engine");
}
```

## Performance Considerations

### ONNX Runtime (Recommended for Most Cases)
- ? Single model format works everywhere
- ? Automatic hardware acceleration
- ? Extensive model zoo support
- ? Active development and updates

### CoreML (iOS/macOS Native)
- ? Optimal for Apple devices with Neural Engine
- ? Lower memory footprint for CoreML models
- ? Requires model conversion
- ? Limited to Apple platforms

### When to Use Each Backend

**Use ONNX Runtime when:**
- You need cross-platform compatibility
- You have ONNX models from PyTorch/TensorFlow
- You want a single codebase
- You need the latest model architectures

**Use CoreML when:**
- You're iOS/macOS only
- You have existing CoreML models
- You need absolute maximum performance on Apple Silicon
- You want lowest memory usage on iPhone

**Use ML Kit when (coming soon):**
- You're Android only
- You have existing TFLite models
- You need Google ML Kit's high-level APIs

## Troubleshooting

### Model Loading Fails
```csharp
try
{
    await mlInfer.LoadModelAsync("model.onnx");
}
catch (FileNotFoundException ex)
{
    // Model file not found - check asset path
}
catch (InvalidOperationException ex)
{
    // Model format not supported by backend
    // Try a different backend or convert the model
}
```

### CoreML Compilation Errors
If you get CoreML compilation errors on iOS:
1. Ensure your model is compatible with the iOS version
2. Try converting with newer `coremltools`
3. Use ONNX Runtime with CoreML execution provider instead

### Performance Issues
1. Check that hardware acceleration is enabled:
   ```csharp
   var backend = mlService.Backend;
   Console.WriteLine($"Using backend: {backend}");
   ```

2. Profile your inference:
   ```csharp
   var sw = Stopwatch.StartNew();
   var result = await mlService.RunInferenceAsync(inputs);
   sw.Stop();
   Console.WriteLine($"Inference took: {sw.ElapsedMilliseconds}ms");
   ```

3. Consider batch size and input resolution

## Migration Guide

### From ONNX-Only to Multi-Backend

**Before:**
```csharp
var onnxInfer = new OnnxRuntimeInfer();
await onnxInfer.LoadModelAsync("model.onnx");
```

**After (still works!):**
```csharp
// Same code works, now with automatic platform optimization
var mlInfer = new OnnxRuntimeInfer(); // or use DI
await mlInfer.LoadModelAsync("model.onnx");
```

**After (using platform defaults):**
```csharp
// Use MLPlugin.Default for automatic backend selection
var mlInfer = MLPlugin.Default;
await mlInfer.LoadModelAsync("model.onnx");

// Or with DI
builder.Services.AddMauiML(); // Automatically optimized per platform
```

## Future Roadmap

- [ ] Full TensorFlow Lite support for Android
- [ ] Windows ML native implementation
- [ ] Model format auto-detection
- [ ] Automatic model conversion helpers
- [ ] Performance profiling tools
- [ ] Quantization support
- [ ] Model caching and precompilation
