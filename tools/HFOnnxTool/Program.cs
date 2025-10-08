using Spectre.Console.Cli;
using HFOnnxTool;

var app = new CommandApp();
app.Configure(cfg =>
{
    cfg.SetApplicationName("hfonnx");
    cfg.AddCommand<InspectCommand>("inspect")
        .WithDescription("List files in a Hugging Face model repo.")
        .WithExample("hfonnx inspect --repo sentence-transformers/all-MiniLM-L6-v2");
    cfg.AddCommand<FetchCommand>("fetch")
        .WithDescription("Download (and optionally select) ONNX model from repo.")
        .WithExample("hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 --output ./models");
    cfg.AddCommand<ConvertCommand>("convert")
        .WithDescription("Export a HF model to ONNX via Python optimum CLI.")
        .WithExample("hfonnx convert --repo d4data/biomedical-ner-all --task token-classification --output ./onnx");
});

return await app.RunAsync(args);
