using Plugin.Maui.ML.Configuration;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Contains unit tests for the FileSystemNlpModelConfigProvider class, verifying its behavior when loading NLP model
///     configurations from the file system.
/// </summary>
/// <remarks>
///     These tests cover scenarios such as handling null or whitespace model keys, missing configuration
///     files, and successful loading of valid configuration files. The tests ensure that the provider returns null for
///     invalid or missing inputs and correctly parses configuration data when present.
/// </remarks>
public class FileSystemNlpModelConfigProviderTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public async Task GetConfigAsync_WithNullOrWhitespaceModelKey_ReturnsNull(string? key)
    {
        // Arrange
        using var temp = new TempDir();
        var provider = new FileSystemNlpModelConfigProvider(temp.Path);

        // Act
        var cfg = await provider.GetConfigAsync(key!);

        // Assert
        Assert.Null(cfg);
    }

    [Fact]
    public async Task GetConfigAsync_WhenConfigFileDoesNotExist_ReturnsNull()
    {
        // Arrange
        using var temp = new TempDir();
        var provider = new FileSystemNlpModelConfigProvider(temp.Path);

        // Act
        var cfg = await provider.GetConfigAsync("does_not_exist");

        // Assert
        Assert.Null(cfg);
    }

    [Fact]
    public async Task GetConfigAsync_WhenConfigFileExists_LoadsConfiguration()
    {
        // Arrange
        using var temp = new TempDir();
        const string modelKey = "testModel";
        var filePath = Path.Combine(temp.Path, modelKey + ".config.json");
        const string json = """
                            {
                              "modelType": "ner",
                              "max_position_embeddings": 512,
                              "pad_token_id": 0,
                              "id2label": { "0": "O", "1": "B-DISEASE", "2": "I-DISEASE" },
                              "label2id": { "O": 0, "B-DISEASE": 1, "I-DISEASE": 2 },
                              "specialTokens": { "ClsToken": "[CLS]", "SepToken": "[SEP]", "PadToken": "[PAD]", "MaskToken": "[MASK]" },
                              "default_pooling": "mean"
                            }
                            """;
        await File.WriteAllTextAsync(filePath, json);
        var provider = new FileSystemNlpModelConfigProvider(temp.Path);

        // Act
        var cfg = await provider.GetConfigAsync(modelKey, CancellationToken.None);

        // Assert
        Assert.NotNull(cfg);
        Assert.Equal("ner", cfg.ModelType);
        Assert.Equal(512, cfg.MaxPositionEmbeddings);
        Assert.Equal(0, cfg.PadTokenId);
        Assert.NotNull(cfg.GetOrderedLabels());
        Assert.Equal(["O", "B-DISEASE", "I-DISEASE"], cfg.GetOrderedLabels());
        Assert.Equal("mean", cfg.DefaultPooling);
        Assert.NotNull(cfg.SpecialTokens);
        Assert.Equal("[CLS]", cfg.SpecialTokens!.ClsToken);
    }

    /// <summary>
    ///     Provides a temporary directory that is automatically deleted when disposed.
    /// </summary>
    /// <remarks>
    ///     Use this class to create and manage a temporary directory for intermediate files or data. The
    ///     directory is deleted recursively when the object is disposed. This class is not thread-safe.
    /// </remarks>
    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = Directory.CreateTempSubdirectory("mauiml_cfg_").FullName;

        public void Dispose()
        {
            try { Directory.Delete(Path, true); }
            catch
            {
                /* ignore */
            }
        }
    }
}
