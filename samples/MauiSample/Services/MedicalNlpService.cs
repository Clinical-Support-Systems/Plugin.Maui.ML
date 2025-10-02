using System.Linq;
using System.Text.RegularExpressions;
using MauiSample.Models;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using Plugin.Maui.ML;

namespace MauiSample.Services
{
    public interface IMedicalNlpService
    {
        bool IsInitialized { get; }
        Task InitializeAsync();
        Task<List<MedicalEntity>> ExtractEntitiesAsync(string text);
        Task<float[]> GetSentenceEmbeddingAsync(string text, int? maxSequenceLength = 256, string pooling = "mean");
    }

    public partial class MedicalNlpService : IMedicalNlpService
    {
        private const string LabelsAssetFile = "ner_labels.txt"; // one label per line matching model output order
        private const string VocabAssetFile = "ner_vocab.txt";
        private const int ModelMaxSequenceLength = 512; // DistilBERT config max_position_embeddings
        private const bool DebugLogging = true;
        private const string CLS = "[CLS]";
        private const string SEP = "[SEP]";
        private const string PAD = "[PAD]";
        private static readonly HashSet<string> _specialTokens = [CLS, SEP, PAD, "[MASK]"];

        private readonly IMLInfer _mlInfer;

        // Default minimal labels (fallback) - real model may have many more
        private string[] _entityLabels =
        [
            "O",
            "B-DISEASE",
            "I-DISEASE",
            "B-DRUG",
            "I-DRUG",
            "B-SYMPTOM",
            "I-SYMPTOM",
            "B-PROCEDURE",
            "I-PROCEDURE"
        ];

        private Tokenizer? _tokenizer;

        public MedicalNlpService(IMLInfer mlInfer)
        {
            _mlInfer = mlInfer;
        }

        public bool IsInitialized { get; private set; }

        public async Task InitializeAsync()
        {
            if (IsInitialized) return;
            try
            {
                await InitializeTokenizerAsync();
                await LoadLabelsAsync();
                await using var modelStream = await FileSystem.OpenAppPackageFileAsync("BiomedicalNER.onnx");
                await _mlInfer.LoadModelAsync(modelStream);

                // Warmup (small dummy) to reduce first-call latency
                try
                {
                    var warmInputIds =
                        new DenseTensor<long>(new long[] { 101, 102 },
                            [1, 2]); // if vocab matches BERT style ids (safe no-op if ignored)
                    var warmMask = new DenseTensor<long>(new long[] { 1, 1 }, [1, 2]);
                    var warmInputs = new Dictionary<string, Tensor<long>>
                    {
                        ["input_ids"] = warmInputIds,
                        ["attention_mask"] = warmMask
                    };
                    _ = _mlInfer.RunInferenceLongInputsAsync(warmInputs); // fire & forget
                }
                catch
                {
                    /* ignore warmup failures */
                }

                if (DebugLogging)
                {
                    var inputs = _mlInfer.GetInputMetadata();
                    var outputs = _mlInfer.GetOutputMetadata();
                    Console.WriteLine(
                        $"[MedicalNLP] Loaded NER model. Inputs: {string.Join(", ", inputs.Keys)}, Outputs: {string.Join(", ", outputs.Keys)}");
                    Console.WriteLine($"[MedicalNLP] Label count: {_entityLabels.Length}");
                }

                IsInitialized = true;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to initialize medical NLP service: {ex.Message}", ex);
            }
        }

        public async Task<List<MedicalEntity>> ExtractEntitiesAsync(string text)
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Service not initialized. Call InitializeAsync first.");
            if (string.IsNullOrWhiteSpace(text)) return [];
            try
            {
                // Tokenize (uncased model) – tokenizer already set to lowercase in initialization
                var encoding = _tokenizer!.EncodeToTokens(text, out _).ToList();
                var tokenIds = encoding.ConvertAll(x => x.Id);
                var tokens = encoding.ConvertAll(x => x.Value).ToList();
                if (tokens.Count == 0) return [];

                // Add special tokens if absent
                if (tokens[0] != CLS)
                {
                    tokens.Insert(0, CLS);
                    tokenIds.Insert(0, FindTokenId(CLS));
                }

                if (tokens[^1] != SEP)
                {
                    if (tokens.Count + 1 <= ModelMaxSequenceLength)
                    {
                        tokens.Add(SEP);
                        tokenIds.Add(FindTokenId(SEP));
                    }
                }

                // Truncate if exceeds max (reserve last token for SEP)
                if (tokenIds.Count > ModelMaxSequenceLength)
                {
                    tokenIds = [.. tokenIds.Take(ModelMaxSequenceLength)];
                    tokens = [.. tokens.Take(ModelMaxSequenceLength)];
                    if (tokens[^1] != SEP)
                    {
                        // force SEP at end if truncated
                        tokenIds[^1] = FindTokenId(SEP);
                        tokens[^1] = SEP;
                    }
                }

                // Pad to max length for stable shape (optional but matches training typical approach)
                var attentionMask = new long[ModelMaxSequenceLength];
                var inputIdsArr = new long[ModelMaxSequenceLength];
                for (var i = 0; i < tokenIds.Count; i++)
                {
                    inputIdsArr[i] = tokenIds[i];
                    attentionMask[i] = 1; // real token
                }

                for (var i = tokenIds.Count; i < ModelMaxSequenceLength; i++)
                {
                    inputIdsArr[i] = FindTokenId(PAD); // pad id (0 typical)
                    attentionMask[i] = 0;
                    tokens.Add(PAD); // keep alignment – ProcessNerOutputs will skip
                }

                var inputIdsTensor = new DenseTensor<long>(inputIdsArr, [1, ModelMaxSequenceLength]);
                var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, ModelMaxSequenceLength]);

                var inputs = new Dictionary<string, Tensor<long>>
                {
                    ["input_ids"] = inputIdsTensor,
                    ["attention_mask"] = attentionMaskTensor
                };

                var outputs = await _mlInfer.RunInferenceLongInputsAsync(inputs);
                var logits = SelectLogitsOutput(outputs);
                if (logits == null)
                {
                    if (DebugLogging) Console.WriteLine("[MedicalNLP] Could not identify logits output tensor.");
                    return [];
                }

                if (DebugLogging)
                {
                    var dimsArr = logits.Dimensions;
                    Console.WriteLine(
                        $"[MedicalNLP] Logits shape: {string.Join("x", dimsArr.ToArray().Select(x => x.ToString()))}; tokens(original)={encoding.Count}");
                }

                return ProcessNerOutputs(logits, [.. tokens], attentionMask);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to extract entities: {ex.Message}", ex);
            }
        }

        public async Task<float[]> GetSentenceEmbeddingAsync(string text, int? maxSequenceLength = 256,
            string pooling = "mean")
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Service not initialized. Call InitializeAsync first.");
            if (!_mlInfer.IsModelLoaded) await _mlInfer.LoadModelFromAssetAsync("SentenceEmbedding.onnx");

            var encoding = _tokenizer!.EncodeToTokens(text, out _).ToList();
            var tokenIds = encoding.ConvertAll(t => (long)t.Id);
            var tokens = encoding.ConvertAll(t => t.Value);

            // Add special tokens
            if (tokens[0] != CLS)
            {
                tokens.Insert(0, CLS);
                tokenIds.Insert(0, FindTokenId(CLS));
            }

            if (tokens[^1] != SEP)
            {
                tokens.Add(SEP);
                tokenIds.Add(FindTokenId(SEP));
            }

            var targetMax = maxSequenceLength.HasValue
                ? Math.Min(ModelMaxSequenceLength, maxSequenceLength.Value)
                : ModelMaxSequenceLength;
            if (tokenIds.Count > targetMax)
            {
                tokenIds = [.. tokenIds.Take(targetMax)];
                tokens = [.. tokens.Take(targetMax)];
                if (tokens[^1] != SEP)
                {
                    tokenIds[^1] = FindTokenId(SEP);
                    tokens[^1] = SEP;
                }
            }

            // Pad
            var inputIdsArr = new long[targetMax];
            var attentionMaskArr = new long[targetMax];
            for (var i = 0; i < tokenIds.Count; i++)
            {
                inputIdsArr[i] = tokenIds[i];
                attentionMaskArr[i] = 1;
            }

            for (var i = tokenIds.Count; i < targetMax; i++)
            {
                inputIdsArr[i] = FindTokenId(PAD);
                attentionMaskArr[i] = 0;
            }

            var inputIdsTensor = new DenseTensor<long>(inputIdsArr, [1, targetMax]);
            var attentionMaskTensor = new DenseTensor<long>(attentionMaskArr, [1, targetMax]);
            var inputs = new Dictionary<string, Tensor<long>>
            {
                ["input_ids"] = inputIdsTensor,
                ["attention_mask"] = attentionMaskTensor
            };
            var outputs = await _mlInfer.RunInferenceLongInputsAsync(inputs);
            Tensor<float> embeddingSource;
            if (outputs.TryGetValue("pooler_output", out var pooler)) embeddingSource = pooler;
            else if (outputs.TryGetValue("last_hidden_state", out var lastHidden)) embeddingSource = lastHidden;
            else embeddingSource = outputs.First().Value;

            var dims = embeddingSource.Dimensions;
            float[] result;
            switch (dims.Length)
            {
                case 2:
                    {
                        var hidden = dims[1];
                        result = new float[hidden];
                        for (var i = 0; i < hidden; i++) result[i] = embeddingSource[0, i];
                        break;
                    }

                case 3:
                    {
                        var seq = dims[1];
                        var hidden = dims[2];
                        var validTokenIndices = Enumerable.Range(0, seq).Where(i => attentionMaskArr[i] == 1).ToArray();
                        var validCount = validTokenIndices.Length;
                        result = new float[hidden];
                        switch (pooling.ToLowerInvariant())
                        {
                            case "cls":
                                for (var h = 0; h < hidden; h++) result[h] = embeddingSource[0, 0, h];
                                break;

                            case "max":
                                for (var h = 0; h < hidden; h++)
                                {
                                    var maxVal = float.MinValue;
                                    foreach (var ti in validTokenIndices)
                                    {
                                        maxVal = Math.Max(maxVal, embeddingSource[0, ti, h]);
                                    }

                                    result[h] = maxVal;
                                }

                                break;

                            default:
                                for (var h = 0; h < hidden; h++)
                                {
                                    double sum = 0;
                                    foreach (var ti in validTokenIndices)
                                    {
                                        sum += embeddingSource[0, ti, h];
                                    }

                                    result[h] = (float)(sum / validCount);
                                }

                                break;
                        }

                        break;
                    }

                default:
                    throw new NotSupportedException($"Unexpected embedding tensor rank: {dims.Length}");
            }

            return result;
        }

        private async Task InitializeTokenizerAsync()
        {
            try
            {
                var vocabPath = Path.Combine(FileSystem.AppDataDirectory, VocabAssetFile);
                if (!File.Exists(vocabPath))
                {
                    await using var stream = await FileSystem.OpenAppPackageFileAsync(VocabAssetFile);
                    await using var fileStream = File.Create(vocabPath);
                    await stream.CopyToAsync(fileStream);
                }

                // DistilBERT base uncased -> lowercase
                var bertOptions = new BertOptions
                {
                    LowerCaseBeforeTokenization = true
                };
                _tokenizer = await BertTokenizer.CreateAsync(vocabPath, bertOptions);
                if (DebugLogging)
                    Console.WriteLine("[MedicalNLP] Loaded tokenizer (lowercased) from " + VocabAssetFile);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to load tokenizer from {VocabAssetFile}: {ex.Message}\nEnsure the file is in Resources/Raw/",
                    ex);
            }
        }

        private async Task LoadLabelsAsync()
        {
            try
            {
                // Try to copy labels file to writable path if exists in assets
                var labelsPath = Path.Combine(FileSystem.AppDataDirectory, LabelsAssetFile);
                if (!File.Exists(labelsPath))
                {
                    try
                    {
                        await using var stream = await FileSystem.OpenAppPackageFileAsync(LabelsAssetFile);
                        await using var fileStream = File.Create(labelsPath);
                        await stream.CopyToAsync(fileStream);
                    }
                    catch
                    {
                        if (DebugLogging)
                        {
                            Console.WriteLine(
                                $"[MedicalNLP] Labels asset '{LabelsAssetFile}' not found. Using fallback labels ({_entityLabels.Length}).");
                        }

                        return; // keep fallback
                    }
                }

                var lines = await File.ReadAllLinesAsync(labelsPath);
                var loaded = lines.Select(l => l.Trim()).Where(l => l.Length > 0 && !l.StartsWith('#')).ToArray();
                if (loaded.Length > 0)
                {
                    _entityLabels = loaded;
                    if (DebugLogging)
                        Console.WriteLine($"[MedicalNLP] Loaded {_entityLabels.Length} labels from {LabelsAssetFile}.");
                }
            }
            catch (Exception ex)
            {
                if (DebugLogging)
                    Console.WriteLine($"[MedicalNLP] Failed loading labels: {ex.Message}; using fallback.");
            }
        }

        private static Tensor<float>? SelectLogitsOutput(Dictionary<string, Tensor<float>> outputs)
        {
            foreach (var kv in outputs)
            {
                if (kv.Key.Contains("logits", StringComparison.OrdinalIgnoreCase)) return kv.Value;
            }

            foreach (var kv in outputs)
            {
                var dims = kv.Value.Dimensions.ToArray();
                if (dims.Length == 3) return kv.Value; // accept first rank-3 now (label count may be large)
            }

            foreach (var kv in outputs)
            {
                var dims = kv.Value.Dimensions.ToArray();
                if (dims.Length == 2) return ReshapeRank2ToRank3(kv.Value);
            }

            return outputs.FirstOrDefault().Value;
        }

        private static Tensor<float> ReshapeRank2ToRank3(Tensor<float> t)
        {
            var dims = t.Dimensions;
            if (dims.Length != 2) return t;
            var reshaped = new DenseTensor<float>([1, dims[0], dims[1]]); // assume [seq, labels]
            for (var i = 0; i < dims[0]; i++)
            {
                for (var j = 0; j < dims[1]; j++)
                {
                    reshaped[0, i, j] = t[i, j];
                }
            }

            return reshaped;
        }

        private static string NormalizeLabel(string raw)
        {
            return string.IsNullOrEmpty(raw) ? raw :
                // Clean bracket artifacts present in some labels
                BracketArtifactRegex().Replace(raw, string.Empty);
        }

        private int FindTokenId(string token)
        {
            var tokens = _tokenizer!.EncodeToTokens(token, out _);
            var first = tokens.FirstOrDefault();
            // EncodedToken may be default(struct) with Id==0 if not found
            return first.Id;
        }

        private List<MedicalEntity> ProcessNerOutputs(Tensor<float> logits, string[] tokens,
            long[]? attentionMask = null)
        {
            var entities = new List<MedicalEntity>();
            var dims = logits.Dimensions;
            if (dims.Length != 3)
            {
                if (DebugLogging) Console.WriteLine($"[MedicalNLP] Unexpected logits rank {dims.Length}.");
                return entities;
            }

            var seqLen = Math.Min(dims[1], tokens.Length);
            var numLabels = dims[2];
            if (DebugLogging)
            {
                Console.WriteLine(
                    $"[MedicalNLP] Process seqLen={seqLen}, numLabels(model)={numLabels}, labelList={_entityLabels.Length}");
            }

            MedicalEntity? currentEntity = null;
            var probs = new float[numLabels];

            for (var i = 0; i < seqLen; i++)
            {
                // Skip padding positions if attention mask provided
                if (attentionMask != null && attentionMask.Length > i && attentionMask[i] == 0)
                    break; // reached padding
                var token = tokens[i];
                if (_specialTokens.Contains(token)) continue; // do not label special tokens

                var maxLogit = float.MinValue;
                for (var j = 0; j < numLabels; j++)
                {
                    var v = logits[0, i, j];
                    probs[j] = v;
                    if (v > maxLogit) maxLogit = v;
                }

                double sum = 0;
                for (var j = 0; j < numLabels; j++)
                {
                    var e = Math.Exp(probs[j] - maxLogit);
                    probs[j] = (float)e;
                    sum += e;
                }

                for (var j = 0; j < numLabels; j++) probs[j] /= (float)sum;
                var bestIdx = 0;
                var best = -1f;
                for (var j = 0; j < numLabels; j++)
                {
                    if (!(probs[j] > best))
                        continue;

                    best = probs[j];
                    bestIdx = j;
                }

                var rawLabel = bestIdx < _entityLabels.Length ? _entityLabels[bestIdx] : "O";
                if (DebugLogging)
                {
                    Console.WriteLine(
                        $"[MedicalNLP] token[{i}]='{token}' -> {rawLabel} ({best:0.000}) (rawIdx={bestIdx})");
                }

                if (rawLabel.StartsWith("B-"))
                {
                    if (currentEntity != null) entities.Add(currentEntity);
                    currentEntity = new MedicalEntity
                    {
                        Text = token.Replace("##", string.Empty),
                        EntityType = NormalizeLabel(rawLabel[2..]),
                        Confidence = best,
                        StartPosition = i,
                        EndPosition = i
                    };
                }
                else if (rawLabel.StartsWith("I-") &&
                         currentEntity != null &&
                         NormalizeLabel(rawLabel[2..]) == currentEntity.EntityType)
                {
                    currentEntity.Text += token.StartsWith("##") ? token[2..] : " " + token;
                    currentEntity.EndPosition = i;
                    currentEntity.Confidence =
                        Math.Max(currentEntity.Confidence, best); // keep max confidence across span
                }
                else
                {
                    if (currentEntity != null)
                    {
                        entities.Add(currentEntity);
                        currentEntity = null;
                    }
                }
            }

            if (currentEntity != null) entities.Add(currentEntity);
            foreach (var e in entities)
            {
                e.Text = e.Text.Replace(" ##", "").Replace("##", "");
            }

            if (DebugLogging)
            {
                Console.WriteLine($"[MedicalNLP] Extracted {entities.Count} entities.");
                foreach (var e in entities)
                {
                    Console.WriteLine($"[MedicalNLP] -> {e.EntityType}: '{e.Text}' ({e.Confidence:0.000})");
                }
            }

            return entities;
        }

        [GeneratedRegex(@"[\[\]\(\)]")]
        private static partial Regex BracketArtifactRegex();
    }
}
