using System.Text.Json;
using System.Text.Json.Serialization;
using System.Linq;

namespace Plugin.Maui.ML.Configuration;

/// <summary>
/// Minimal NLP model configuration for runtime inference.
/// </summary>
public sealed class NlpModelConfig
{
    public string? ModelType { get; set; }

    [JsonPropertyName("max_position_embeddings")] public int? MaxPositionEmbeddings { get; set; }

    [JsonPropertyName("pad_token_id")] public int? PadTokenId { get; set; }

    [JsonPropertyName("id2label")] public Dictionary<string, string>? Id2LabelRaw { get; set; }

    [JsonPropertyName("label2id")] public Dictionary<string, int>? Label2Id { get; set; }

    public SpecialTokenConfig? SpecialTokens { get; set; }

    public LabelNormalization? Normalization { get; set; }

    [JsonPropertyName("default_pooling")] public string? DefaultPooling { get; set; }

    [JsonIgnore]
    public string[]? OrderedLabels => Id2LabelRaw == null
        ? null
        : Id2LabelRaw
            .Select(kv => (Ok: int.TryParse(kv.Key, out var i), i, kv.Value))
            .Where(t => t.Ok)
            .OrderBy(t => t.i)
            .Select(t => t.Value)
            .ToArray();

    public static async Task<NlpModelConfig?> LoadAsync(Stream jsonStream, CancellationToken ct = default)
    {
        var opts = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
            NumberHandling = JsonNumberHandling.AllowReadingFromString
        };
        return await JsonSerializer.DeserializeAsync<NlpModelConfig>(jsonStream, opts, ct);
    }

    public sealed class SpecialTokenConfig
    {
        public string? ClsToken { get; set; } = "[CLS]";
        public string? SepToken { get; set; } = "[SEP]";
        public string? PadToken { get; set; } = "[PAD]";
        public string? MaskToken { get; set; } = "[MASK]";
    }

    public sealed class LabelNormalization
    {
        public Dictionary<string, string>? ReplaceMap { get; set; }
        public string[]? StripPatterns { get; set; }
    }
}
