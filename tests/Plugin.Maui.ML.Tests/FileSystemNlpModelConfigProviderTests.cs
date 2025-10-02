using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Plugin.Maui.ML.Configuration;
using Xunit;

namespace Plugin.Maui.ML.Tests;

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
        var modelKey = "testModel";
        var filePath = Path.Combine(temp.Path, modelKey + ".config.json");
        var json = """
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
        Assert.Equal("ner", cfg!.ModelType);
        Assert.Equal(512, cfg.MaxPositionEmbeddings);
        Assert.Equal(0, cfg.PadTokenId);
        Assert.NotNull(cfg.GetOrderedLabels());
        Assert.Equal(["O", "B-DISEASE", "I-DISEASE"], cfg.GetOrderedLabels());
        Assert.Equal("mean", cfg.DefaultPooling);
        Assert.NotNull(cfg.SpecialTokens);
        Assert.Equal("[CLS]", cfg.SpecialTokens!.ClsToken);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = Directory.CreateTempSubdirectory("mauiml_cfg_").FullName;
        public void Dispose()
        {
            try { Directory.Delete(Path, true); } catch { /* ignore */ }
        }
    }
}
