using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using Plugin.Maui.ML;
using Plugin.Maui.ML.Utilities;

// Create a host with dependency injection
var builder = Host.CreateApplicationBuilder(args);

// Configure logging
builder.Logging.AddConsole();

// Register ML services
builder.Services.AddMauiML(config =>
{
    config.EnablePerformanceLogging = true;
    config.UseTransientService = false; // Use singleton for better performance
});

var host = builder.Build();

// Get the ML inference service
var mlService = host.Services.GetRequiredService<IMLInfer>();
var logger = host.Services.GetRequiredService<ILogger<Program>>();

logger.LogInformation("Plugin.Maui.ML Console Sample Started");

try
{
    // Demo: Create sample tensors for inference
    // This would normally come from your actual data (images, text, etc.)
    var inputData = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
    var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 4 });

    var inputs = new Dictionary<string, Tensor<float>>
    {
        ["input"] = inputTensor
    };

    logger.LogInformation("Created sample input tensor with shape [1, 4]");
    logger.LogInformation("Input data: [{data}]", string.Join(", ", inputData));

    // Note: This will fail without an actual model file
    // In a real scenario, you would load a model first:
    // await mlService.LoadModelAsync("path/to/your/model.onnx");

    logger.LogInformation("Model loading and inference would happen here");
    logger.LogInformation("To use this plugin:");
    logger.LogInformation("1. Place your .onnx model file in the application directory");
    logger.LogInformation("2. Call await mlService.LoadModelAsync(\"your-model.onnx\")");
    logger.LogInformation("3. Create input tensors matching your model's requirements");
    logger.LogInformation("4. Call await mlService.RunInferenceAsync(inputs)");

    // Demonstrate utility functions
    logger.LogInformation("Demonstrating tensor utilities:");

    // Show tensor shape
    var shapeString = TensorHelper.GetShapeString(inputTensor);
    logger.LogInformation("Tensor shape: {shape}", shapeString);

    // Show normalization
    var normalizedTensor = TensorHelper.Normalize(inputTensor);
    var normalizedData = TensorHelper.ToArray(normalizedTensor);
    logger.LogInformation("Normalized data: [{data}]", string.Join(", ", normalizedData.Select(x => x.ToString("F3"))));

    // Show softmax
    var softmaxTensor = TensorHelper.Softmax(inputTensor);
    var softmaxData = TensorHelper.ToArray(softmaxTensor);
    logger.LogInformation("Softmax data: [{data}]", string.Join(", ", softmaxData.Select(x => x.ToString("F3"))));

    logger.LogInformation("Sample completed successfully!");
}
catch (Exception ex)
{
    logger.LogError(ex, "An error occurred during the sample execution");
}
finally
{
    // Clean up
    if (mlService is IDisposable disposable)
    {
        disposable.Dispose();
    }

    logger.LogInformation("Plugin.Maui.ML Console Sample Ended");
}
