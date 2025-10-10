using System.Text;
using Plugin.Maui.ML.Configuration;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Provides unit tests for the NlpModelConfig class, verifying configuration deserialization, label ordering, and
///     default values for special tokens.
/// </summary>
/// <remarks>
///     These tests ensure that NlpModelConfig correctly handles various JSON formats, including
///     case-insensitive property names, comments, trailing commas, and numeric values represented as strings. The tests
///     also validate behavior for edge cases such as null or empty label dictionaries and confirm that cancellation is
///     respected during asynchronous loading.
/// </remarks>
public class NlpModelConfigTests
{
    [Fact]
    public void GetOrderedLabels_WhenId2LabelRawNull_ReturnsNull()
    {
        var cfg = new NlpModelConfig
        {
            Id2LabelRaw = null
        };
        var result = cfg.GetOrderedLabels();
        Assert.Null(result);
    }

    [Fact]
    public void GetOrderedLabels_WithEmptyDictionary_ReturnsEmptyArray()
    {
        var cfg = new NlpModelConfig
        {
            Id2LabelRaw = new Dictionary<string, string>()
        };
        var result = cfg.GetOrderedLabels();
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void GetOrderedLabels_WithOnlyNonIntegerKeys_ReturnsEmptyArray()
    {
        var cfg = new NlpModelConfig
        {
            Id2LabelRaw = new Dictionary<string, string>
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
            Id2LabelRaw = new Dictionary<string, string>
            {
                ["2"] = "C",
                ["x"] = "Ignored",
                ["0"] = "A",
                ["1"] = "B"
            }
        };
        var result = cfg.GetOrderedLabels();
        Assert.Equal(["A", "B", "C"], result);
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
        const string json = """
                            {
                              "modelType": "ner",
                              "max_position_embeddings": 256,
                              "pad_token_id": 5
                            }
                            """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal("ner", cfg.ModelType);
        Assert.Equal(256, cfg.MaxPositionEmbeddings);
        Assert.Equal(5, cfg.PadTokenId);
    }

    [Fact]
    public async Task LoadAsync_SupportsCaseInsensitivePropertyNames()
    {
        const string json = """
                            {
                              "MODELTYPE": "sequence_classification"
                            }
                            """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal("sequence_classification", cfg.ModelType);
    }

    [Fact]
    public async Task LoadAsync_SupportsCommentsAndTrailingCommas()
    {
        const string json = """
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
        Assert.Equal("ner", cfg.ModelType);
        Assert.Equal(512, cfg.MaxPositionEmbeddings);
        Assert.Equal(0, cfg.PadTokenId);
        Assert.Equal(["O", "B"], cfg.GetOrderedLabels());
    }

    [Fact]
    public async Task LoadAsync_AllowsNumbersAsStrings()
    {
        const string json = """
                            {
                              "modelType": "ner",
                              "max_position_embeddings": "128",
                              "pad_token_id": "42"
                            }
                            """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal(128, cfg.MaxPositionEmbeddings);
        Assert.Equal(42, cfg.PadTokenId);
    }

    [Fact]
    public async Task LoadAsync_CanOrderLabelsAfterDeserialization()
    {
        const string json = """
                            {
                              "modelType": "ner",
                              "id2label": { "2":"C", "0":"A", "1":"B", "z":"Ignored" }
                            }
                            """;
        using var stream = CreateStreamFromString(json);
        var cfg = await NlpModelConfig.LoadAsync(stream);
        Assert.NotNull(cfg);
        Assert.Equal(["A", "B", "C"], cfg.GetOrderedLabels());
    }

    [Fact]
    public async Task LoadAsync_CancellationBeforeStart_Throws()
    {
        const string json = """ { "modelType": "ner" } """;
        using var stream = CreateStreamFromString(json);
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await NlpModelConfig.LoadAsync(stream, cts.Token);
        });
    }

    private static MemoryStream CreateStreamFromString(string s)
    {
        return new MemoryStream(Encoding.UTF8.GetBytes(s));
    }
}
