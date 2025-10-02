using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Plugin.Maui.ML.Configuration;
using Xunit;

namespace Plugin.Maui.ML.Tests;

public class NlpModelConfigTests
{
    [Fact]
    public void GetOrderedLabels_WhenId2LabelRawNull_ReturnsNull()
    {
        var cfg = new NlpModelConfig { Id2LabelRaw = null };
        var result = cfg.GetOrderedLabels();
        Assert.Null(result);
    }

    [Fact]
    public void GetOrderedLabels_WithEmptyDictionary_ReturnsEmptyArray()
    {
        var cfg = new NlpModelConfig { Id2LabelRaw = new() };
        var result = cfg.GetOrderedLabels();
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void GetOrderedLabels_WithOnlyNonIntegerKeys_ReturnsEmptyArray()
    {
        var cfg = new NlpModelConfig
        {
            Id2LabelRaw = new()
            {
                ["A"] = "Alpha",
                ["B"] = "Beta"
            }
        };
        var result = cfg.GetOrderedLabels();
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void GetOrderedLabels_MixedIntegerAndNonInteger_SortsAndFilters()
    {
        var cfg = new NlpModelConfig
        {
            Id2LabelRaw = new()
            {
                ["2"] = "C",
                ["x"] = "Ignored",
                ["0"] = "A",
                ["1"] = "B"
            }
        };
        var result = cfg.GetOrderedLabels();
        Assert.Equal(new[] { "A", "B", "C" }, result);
    }

    [Fact]
    public void SpecialTokenConfig_DefaultValues()
    {
        var st = new NlpModelConfig.SpecialTokenConfig();
        Assert.Equal("[CLS]", st.ClsToken);
        Assert.Equal("[SEP]", st.SepToken);
        Assert.Equal("[PAD]", st.PadToken);
        Assert.Equal("[MASK]", st.MaskToken);
    }

    [Fact]
    public async Task LoadAsync_DeserializesBasicConfig()
    {
        var json = """
        {
          "modelType": "ner",
          "max_position_embeddings": 256,
          "pad_token_id": 5
        }
        """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal("ner", cfg!.ModelType);
        Assert.Equal(256, cfg.MaxPositionEmbeddings);
        Assert.Equal(5, cfg.PadTokenId);
    }

    [Fact]
    public async Task LoadAsync_SupportsCaseInsensitivePropertyNames()
    {
        var json = """
        {
          "MODELTYPE": "sequence_classification"
        }
        """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal("sequence_classification", cfg!.ModelType);
    }

    [Fact]
    public async Task LoadAsync_SupportsCommentsAndTrailingCommas()
    {
        var json = """
        {
          // comment line
          "modelType": "ner",
          "max_position_embeddings": 512,
          "pad_token_id": 0,
          "id2label": { "0": "O", "1": "B", }, // trailing comma
          "label2id": { "O": 0, "B": 1, }, // trailing
        }
        """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal("ner", cfg!.ModelType);
        Assert.Equal(512, cfg.MaxPositionEmbeddings);
        Assert.Equal(0, cfg.PadTokenId);
        Assert.Equal(new[] { "O", "B" }, cfg.GetOrderedLabels());
    }

    [Fact]
    public async Task LoadAsync_AllowsNumbersAsStrings()
    {
        var json = """
        {
          "modelType": "ner",
          "max_position_embeddings": "128",
          "pad_token_id": "42"
        }
        """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal(128, cfg!.MaxPositionEmbeddings);
        Assert.Equal(42, cfg.PadTokenId);
    }

    [Fact]
    public async Task LoadAsync_CanOrderLabelsAfterDeserialization()
    {
        var json = """
        {
          "modelType": "ner",
          "id2label": { "2":"C", "0":"A", "1":"B", "z":"Ignored" }
        }
        """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal(new[] { "A", "B", "C" }, cfg!.GetOrderedLabels());
    }

    [Fact]
    public async Task LoadAsync_CancellationBeforeStart_Throws()
    {
        var json = """ { "modelType": "ner" } """;
        using var stream = CreateStreamFromString(json);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await NlpModelConfig.LoadAsync(stream, cts.Token);
        });
    }

    private static MemoryStream CreateStreamFromString(string s)
        => new(System.Text.Encoding.UTF8.GetBytes(s));
}
