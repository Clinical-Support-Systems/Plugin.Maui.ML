using System.Text.Json;
using System.Text.Json.Serialization;

namespace Plugin.Maui.ML.Configuration;

/// <summary>
///     Minimal Natural Language Processing (NLP) model configuration for runtime inference.
/// </summary>
public sealed class NlpModelConfig
{
    /// <summary>
    ///     Provides a cached set of default options for JSON serialization and deserialization operations.
    /// </summary>
    /// <remarks>
    ///     The options enable case-insensitive property matching, skip comments in JSON input, allow
    ///     trailing commas, and permit reading numbers from string values. The type information resolver is set to use the
    ///     default context for NLP model types. Use this instance to ensure consistent and efficient JSON processing across
    ///     the application.
    /// </remarks>
    private static readonly JsonSerializerOptions CachedJsonSerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        NumberHandling = JsonNumberHandling.AllowReadingFromString,
        TypeInfoResolver = NlpModelJsonContext.Default
    };

    /// <summary>
    ///     Gets or sets the type of model used for processing or inference.
    /// </summary>
    public string? ModelType { get; set; }

    /// <summary>
    ///     Gets or sets the maximum number of position embeddings supported by the model.
    /// </summary>
    [JsonPropertyName("max_position_embeddings")]
    public int? MaxPositionEmbeddings { get; set; }

    /// <summary>
    ///     Gets or sets the token ID used for padding sequences during model input processing.
    /// </summary>
    /// <remarks>
    ///     Specify this value to indicate which token should be used to pad input sequences to a uniform
    ///     length. If not set, no padding token will be applied automatically.
    /// </remarks>
    [JsonPropertyName("pad_token_id")]
    public int? PadTokenId { get; set; }

    /// <summary>
    ///     Gets or sets the raw mapping of label identifiers to their corresponding label names as provided in the source
    ///     data.
    /// </summary>
    [JsonPropertyName("id2label")]
    public Dictionary<string, string>? Id2LabelRaw { get; set; }

    /// <summary>
    ///     Gets or sets the mapping of label names to their corresponding integer identifiers.
    /// </summary>
    [JsonPropertyName("label2id")]
    public Dictionary<string, int>? Label2Id { get; set; }

    /// <summary>
    ///     Gets or sets the configuration for special tokens used in processing or generation tasks.
    /// </summary>
    /// <remarks>
    ///     Use this property to specify custom token settings, such as start-of-sequence or
    ///     end-of-sequence tokens, that may affect how input or output is handled. If not set, default token behavior will
    ///     be applied.
    /// </remarks>
    public SpecialTokenConfig? SpecialTokens { get; set; }

    /// <summary>
    ///     Gets or sets the normalization method to apply to label values before processing.
    /// </summary>
    /// <remarks>
    ///     Specify a normalization method to ensure label values are consistently formatted or scaled.
    ///     If no value is set, no normalization will be applied.
    /// </remarks>
    public LabelNormalization? Normalization { get; set; }

    /// <summary>
    ///     Gets or sets the default pooling strategy to be used for connections.
    /// </summary>
    [JsonPropertyName("default_pooling")]
    public string? DefaultPooling { get; set; }

    /// <summary>
    ///     Returns an array of label strings ordered by their associated integer keys.
    /// </summary>
    /// <remarks>
    ///     Labels with non-integer keys are excluded from the result. The returned array may be empty if
    ///     no valid integer-keyed labels exist.
    /// </remarks>
    /// <returns>
    ///     An array of labels sorted by their integer keys in ascending order. Returns <see langword="null" /> if no label
    ///     data is available.
    /// </returns>
    public string[]? GetOrderedLabels()
    {
        return Id2LabelRaw?.Select(kv => (Ok: int.TryParse(kv.Key, out var i), i, kv.Value))
            .Where(t => t.Ok)
            .OrderBy(t => t.i)
            .Select(t => t.Value)
            .ToArray();
    }

    /// <summary>
    ///     Asynchronously loads an instance of <see cref="NlpModelConfig" /> from a JSON stream.
    /// </summary>
    /// <param name="jsonStream">
    ///     The stream containing the JSON representation of the model configuration. The stream must be readable and
    ///     positioned at the start of the JSON data.
    /// </param>
    /// <param name="ct">A cancellation token that can be used to cancel the asynchronous operation.</param>
    /// <returns>
    ///     A task that represents the asynchronous load operation. The task result is an <see cref="NlpModelConfig" />
    ///     instance if the JSON is valid; otherwise, <see langword="null" />.
    /// </returns>
    public static async Task<NlpModelConfig?> LoadAsync(Stream jsonStream, CancellationToken ct = default)
    {
        // Ensure the exact expected exception type for pre-canceled tokens
        if (ct.IsCancellationRequested)
            throw new OperationCanceledException(ct);

        try
        {
            return await JsonSerializer.DeserializeAsync<NlpModelConfig>(jsonStream, CachedJsonSerializerOptions, ct)
                .ConfigureAwait(false);
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            // Normalize TaskCanceledException to OperationCanceledException for test expectations
            throw new OperationCanceledException(ct);
        }
    }

    /// <summary>
    ///     Represents a configuration for special tokens used in natural language processing models, such as
    ///     classification, separation, padding, and masking tokens.
    /// </summary>
    /// <remarks>
    ///     This class is commonly used to specify the string representations of special tokens required
    ///     by transformer-based models and tokenizers. Each property corresponds to a specific token type and can be
    ///     customized to match the requirements of different models or datasets.
    /// </remarks>
    public sealed class SpecialTokenConfig
    {
        public string? ClsToken { get; set; } = "[CLS]";
        public string? SepToken { get; set; } = "[SEP]";
        public string? PadToken { get; set; } = "[PAD]";
        public string? MaskToken { get; set; } = "[MASK]";
    }

    /// <summary>
    ///     Represents configuration options for normalizing label strings, including replacement mappings and patterns to
    ///     strip.
    /// </summary>
    /// <remarks>
    ///     Use this class to specify how label text should be standardized, such as replacing specific
    ///     substrings or removing unwanted patterns. This is typically used in scenarios where consistent label formatting
    ///     is required, such as data preprocessing or user interface normalization.
    /// </remarks>
    public sealed class LabelNormalization
    {
        public Dictionary<string, string>? ReplaceMap { get; set; }
        public string[]? StripPatterns { get; set; }
    }
}

/// <summary>
///     Provides a source-generated JSON serialization context for the NlpModelConfig type, enabling efficient
///     serialization
///     and deserialization using System.Text.Json.
/// </summary>
/// <remarks>
///     This context is configured to always include all properties during serialization and to use
///     metadata-based generation for optimal performance. The context should be used when serializing or deserializing
///     NlpModelConfig instances to ensure compatibility with the expected JSON schema.
/// </remarks>
[JsonSourceGenerationOptions(
    GenerationMode = JsonSourceGenerationMode.Metadata,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    WriteIndented = false)]
[JsonSerializable(typeof(NlpModelConfig))]
internal partial class NlpModelJsonContext : JsonSerializerContext;