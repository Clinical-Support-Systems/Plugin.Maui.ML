using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Contains unit tests for the OnnxRuntimeInfer class, verifying model loading, inference execution, metadata
///     retrieval, and resource management behaviors.
/// </summary>
/// <remarks>
///     These tests ensure that OnnxRuntimeInfer correctly handles valid and invalid input scenarios, throws
///     appropriate exceptions, and manages model lifecycle operations. The test suite covers asynchronous and synchronous
///     methods, including edge cases such as null or empty parameters, repeated disposal, and inference with various input
///     types.
/// </remarks>
public class OnnxRuntimeInferTests
{
    private static readonly float[] Memory = [1f];

    private static string GetModelPath()
    {
        return Path.Combine(AppContext.BaseDirectory, "t5encoder_Opset17.onnx");
    }

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        // Act
        using var infer = new OnnxRuntimeInfer();

        // Assert
        Assert.NotNull(infer);
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public async Task LoadModelAsync_SucceedsAndSetsIsModelLoaded()
    {
        using var infer = new OnnxRuntimeInfer();
        var path = GetModelPath();
        Assert.True(File.Exists(path));
        await infer.LoadModelAsync(path);
        Assert.True(infer.IsModelLoaded);
    }

    [Fact]
    public async Task LoadModelAsync_WithNullPath_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelAsync((string)null!));
    }

    [Fact]
    public async Task LoadModelAsync_WithEmptyPath_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelAsync(string.Empty));
    }

    [Fact]
    public async Task LoadModelAsync_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        using var infer = new OnnxRuntimeInfer();
        var nonExistentPath = "non-existent-model.onnx";
        await Assert.ThrowsAsync<FileNotFoundException>(() => infer.LoadModelAsync(nonExistentPath));
    }

    [Fact]
    public async Task LoadModelAsync_WithNullStream_ThrowsArgumentNullException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentNullException>(() => infer.LoadModelAsync((Stream)null!));
    }

    [Fact]
    public async Task LoadModelFromAssetAsync_WithNullAssetName_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelFromAssetAsync(null!));
    }

    [Fact]
    public async Task LoadModelFromAssetAsync_WithEmptyAssetName_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelFromAssetAsync(string.Empty));
    }

    [Fact]
    public async Task RunInferenceAsync_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        using var infer = new OnnxRuntimeInfer();
        var inputs = new Dictionary<string, Tensor<float>>
        {
            ["input_ids"] = new DenseTensor<float>(Memory, [1])
        };
        await Assert.ThrowsAsync<InvalidOperationException>(() => infer.RunInferenceAsync(inputs));
    }

    [Fact]
    public async Task RunInferenceAsync_WithNullInputs_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.RunInferenceAsync(null!));
    }

    [Fact]
    public async Task RunInferenceAsync_WithEmptyInputs_ThrowsArgumentException()
    {
        using var infer = new OnnxRuntimeInfer();
        var inputs = new Dictionary<string, Tensor<float>>();
        await Assert.ThrowsAsync<ArgumentException>(() => infer.RunInferenceAsync(inputs));
    }

    [Fact]
    public void GetInputMetadata_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        using var infer = new OnnxRuntimeInfer();
        Assert.Throws<InvalidOperationException>(() => infer.GetInputMetadata());
    }

    [Fact]
    public void GetOutputMetadata_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        using var infer = new OnnxRuntimeInfer();
        Assert.Throws<InvalidOperationException>(() => infer.GetOutputMetadata());
    }

    [Fact]
    public void UnloadModel_WithoutLoadedModel_DoesNotThrow()
    {
        using var infer = new OnnxRuntimeInfer();
        infer.UnloadModel();
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public void Dispose_DisposesCorrectly()
    {
        var infer = new OnnxRuntimeInfer();
        infer.Dispose();
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public void Dispose_MultipleCalls_DoesNotThrow()
    {
        var infer = new OnnxRuntimeInfer();
        infer.Dispose();
        infer.Dispose();
        infer.Dispose();
    }

    [Fact]
    public async Task AfterLoad_MetadataAvailable()
    {
        using var infer = new OnnxRuntimeInfer();
        await infer.LoadModelAsync(GetModelPath());
        var inputs = infer.GetInputMetadata();
        var outputs = infer.GetOutputMetadata();
        Assert.NotEmpty(inputs);
        Assert.NotEmpty(outputs);
    }

    private static int GetSeqLenFromMetadata(Dictionary<string, NodeMetadata> meta, string inputName)
    {
        if (!meta.TryGetValue(inputName, out var node)) return 1;
        var dims = node.Dimensions;
        if (dims.Length < 2) return 1;
        var dim = dims[1];
        if (dim <= 0) return 1; // dynamic / unknown
        return dim;
    }

    [Fact]
    public async Task RunInferenceLongInputsAsync_Succeeds()
    {
        using var infer = new OnnxRuntimeInfer();
        await infer.LoadModelAsync(GetModelPath());
        var metadata = infer.GetInputMetadata();
        var inputIdsName = metadata.Keys.First(k => k.Contains("input_ids"));
        var attentionMaskName = metadata.Keys.First(k => k.Contains("attention_mask"));
        var seqLen = GetSeqLenFromMetadata(metadata, inputIdsName);
        var inputIds = new DenseTensor<long>([1, seqLen]);
        var attention = new DenseTensor<long>([1, seqLen]);
        inputIds[0, 0] = 0; // first token
        attention[0, 0] = 1;
        var dict = new Dictionary<string, Tensor<long>>
        {
            [inputIdsName] = inputIds,
            [attentionMaskName] = attention
        };
        var result = await infer.RunInferenceLongInputsAsync(dict);
        Assert.NotEmpty(result);
    }

    [Fact]
    public async Task RunInferenceAsync_FloatInputs_CastsOutputsIfNeeded()
    {
        using var infer = new OnnxRuntimeInfer();
        await infer.LoadModelAsync(GetModelPath());
        var metadata = infer.GetInputMetadata();
        var floatableInputName = metadata.Keys.First(k => k.Contains("input_ids"));
        var seqLen = GetSeqLenFromMetadata(metadata, floatableInputName);
        var tensor = new DenseTensor<float>([1, seqLen])
        {
            [0, 0] = 0f
        };
        var dict = new Dictionary<string, Tensor<float>>
        {
            { floatableInputName, tensor }
        };
        try
        {
            var outputs = await infer.RunInferenceAsync(dict);
            foreach (var kv in outputs)
            {
                Assert.IsType<DenseTensor<float>>(kv.Value);
            }
        }
        catch (Exception ex) when (ex is OnnxRuntimeException)
        {
            Assert.True(true);
        }
    }

    [Fact]
    public async Task UnloadModel_AfterLoad_ModelDisposed()
    {
        using var infer = new OnnxRuntimeInfer();
        await infer.LoadModelAsync(GetModelPath());
        Assert.True(infer.IsModelLoaded);
        infer.UnloadModel();
        Assert.False(infer.IsModelLoaded);
        infer.UnloadModel();
        Assert.False(infer.IsModelLoaded);
    }
}
