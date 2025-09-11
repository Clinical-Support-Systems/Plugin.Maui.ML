# Plugin.Maui.ML

A comprehensive .NET MAUI plugin that provides ONNX runtime inference capabilities with Git LFS support for ML model files.

## Features

- 🧠 **ONNX Runtime Integration**: Full support for ONNX model inference
- 🏗️ **Platform-Specific Optimizations**: Leverages native ML acceleration on each platform
- 📱 **Cross-Platform**: Supports Android, iOS, macOS Catalyst, and Windows
- 🔧 **Dependency Injection**: Easy integration with .NET DI container
- 🛠️ **Utility Classes**: Helper classes for tensor operations and model management
- 📦 **NuGet Package**: Available as a ready-to-use NuGet package
- 🔍 **Git LFS Support**: Proper handling of large ML model files
- 🧪 **Comprehensive Tests**: Full unit test coverage

## Installation

```xml
<PackageReference Include="Plugin.Maui.ML" Version="1.0.0" />
```

## Quick Start

### 1. Register the Service

```csharp
// In your MauiProgram.cs
public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
                fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
            });

        // Register ML services
        builder.Services.AddMauiML(config =>
        {
            config.EnablePerformanceLogging = true;
            config.MaxConcurrentInferences = 2;
        });

        return builder.Build();
    }
}
```

### 2. Use in Your Code

```csharp
public class MainPage : ContentPage
{
    private readonly IMLInfer _mlService;

    public MainPage(IMLInfer mlService)
    {
        _mlService = mlService;
        InitializeComponent();
    }

    private async void OnPredictClicked(object sender, EventArgs e)
    {
        try
        {
            // Load your ONNX model
            await _mlService.LoadModelFromAssetAsync("my-model.onnx");

            // Prepare input tensors
            var inputData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var inputTensor = TensorHelper.CreateTensor(inputData, new int[] { 1, 4 });
            var inputs = new Dictionary<string, Tensor<float>>
            {
                ["input"] = inputTensor
            };

            // Run inference
            var results = await _mlService.RunInferenceAsync(inputs);

            // Process results
            foreach (var output in results)
            {
                var outputData = TensorHelper.ToArray(output.Value);
                Console.WriteLine($"Output '{output.Key}': [{string.Join(", ", outputData)}]");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("Error", ex.Message, "OK");
        }
    }
}
```

## API Reference

### IMLInfer Interface

The main interface for ML inference operations.

#### Methods

- `Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)`
  - Load an ONNX model from a file path

- `Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)`  
  - Load an ONNX model from a stream

- `Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)`
  - Load an ONNX model from MAUI assets

- `Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(Dictionary<string, Tensor<float>> inputs, CancellationToken cancellationToken = default)`
  - Run inference on the loaded model

- `Dictionary<string, NodeMetadata> GetInputMetadata()`
  - Get input metadata for the loaded model

- `Dictionary<string, NodeMetadata> GetOutputMetadata()`
  - Get output metadata for the loaded model

- `void UnloadModel()`
  - Dispose of the loaded model and release resources

#### Properties

- `bool IsModelLoaded { get; }`
  - Check if a model is currently loaded

### TensorHelper Utility Class

Helper utilities for working with tensors.

#### Methods

- `static Tensor<float> CreateTensor(float[] data, int[] dimensions)`
  - Create a tensor from a float array

- `static Tensor<float> CreateTensor(float[,] data)`
  - Create a tensor from a 2D float array

- `static Tensor<float> CreateTensor(float[,,] data)`
  - Create a tensor from a 3D float array

- `static float[] ToArray(Tensor<float> tensor)`
  - Convert tensor to float array

- `static string GetShapeString(Tensor<float> tensor)`
  - Get tensor shape as a string

- `static Tensor<float> Reshape(Tensor<float> tensor, int[] newDimensions)`
  - Reshape a tensor to new dimensions

- `static Tensor<float> Normalize(Tensor<float> tensor)`
  - Normalize tensor values to 0-1 range

- `static Tensor<float> Softmax(Tensor<float> tensor)`
  - Apply softmax function to tensor

### Configuration

Configure the ML services with `MLConfiguration`:

```csharp
builder.Services.AddMauiML(config =>
{
    config.UseTransientService = false; // Use singleton (default)
    config.EnablePerformanceLogging = true;
    config.MaxConcurrentInferences = Environment.ProcessorCount;
    config.DefaultModelAssetPath = "models/default-model.onnx";
});
```

#### Configuration Options

- `UseTransientService`: Whether to use transient service lifetime (default: false, uses singleton)
- `EnablePerformanceLogging`: Enable performance logging (default: false)  
- `MaxConcurrentInferences`: Maximum number of concurrent inference operations (default: processor count)
- `DefaultModelAssetPath`: Default model asset path (default: null)

## Platform-Specific Features

### Android
- **NNAPI Support**: Automatic Neural Network API acceleration on supported devices
- **Asset Loading**: Load models directly from Android assets folder

### iOS/macOS
- **CoreML Integration**: Automatic CoreML acceleration when available
- **Neural Engine**: Support for Apple Neural Engine on compatible devices
- **Bundle Resources**: Load models from iOS/macOS app bundles

### Windows
- **DirectML Support**: Hardware acceleration via DirectML on compatible GPUs
- **Package Resources**: Load models from Windows app packages

## Git LFS Configuration

The plugin is configured to work with Git LFS for handling large model files. The following file types are automatically tracked:

- `*.onnx` - ONNX model files
- `*.pb` - Protocol Buffer model files  
- `*.tflite` - TensorFlow Lite models
- `*.h5` - HDF5 model files
- `*.pkl` - Pickle model files
- `*.model` - Generic model files
- `*.bin` - Binary model files
- `*.weights` - Model weights files

## Examples

Check out the sample projects in the `samples/` directory:

- **ConsoleApp**: Basic console application demonstrating plugin usage
- Comprehensive examples showing different model types and use cases

## Testing

Run the comprehensive test suite:

```bash
dotnet test tests/Plugin.Maui.ML.Tests/
```

The test suite includes:
- Unit tests for all core functionality
- Integration tests for platform-specific features
- Performance benchmarks
- Memory leak detection

## Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/Clinical-Support-Systems/Plugin.Maui.ML.git
   ```

2. Restore dependencies:
   ```bash
   dotnet restore
   ```

3. Build the solution:
   ```bash
   dotnet build
   ```

4. Run tests:
   ```bash
   dotnet test
   ```

5. Create NuGet package:
   ```bash
   dotnet pack src/Plugin.Maui.ML/Plugin.Maui.ML.csproj
   ```

## Performance Considerations

- Use singleton service lifetime for better performance (default)
- Consider using platform-specific execution providers for optimal performance
- Monitor memory usage when working with large models
- Use async methods to avoid blocking the UI thread

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure the model file path is correct
   - Verify the model is a valid ONNX format
   - Check file permissions

2. **Memory Issues**
   - Use `UnloadModel()` when done with a model
   - Consider using transient services for memory-sensitive scenarios
   - Monitor memory usage with large models

3. **Platform-Specific Issues**
   - Verify platform-specific dependencies are installed
   - Check execution provider availability
   - Review platform-specific documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [Microsoft ONNX Runtime](https://onnxruntime.ai/)
- Inspired by the .NET MAUI community plugins
- Thanks to all contributors and users