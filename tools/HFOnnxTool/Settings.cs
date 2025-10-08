using System.ComponentModel;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HFOnnxTool;

/// <summary>
///     Provides common settings for Hugging Face repository commands, including repository location, revision, and
///     authentication token.
/// </summary>
/// <remarks>
///     This class serves as a base for command-line operations that interact with Hugging Face repositories.
///     It validates that a repository is specified and supports configuration of the repository revision and access token.
///     The access token can be provided directly or via the HF_TOKEN environment variable.
/// </remarks>
public class HfBaseSettings : CommandSettings
{
    [CommandOption("--repo <REPO>")]
    [Description("Hugging Face repo (org/name) or full URL.")]
    public string? Repo { get; set; } = null!;

    [CommandOption("--revision <REV>")]
    [Description("Git revision (branch/tag/sha). Default=main")]
    [DefaultValue("main")]
    public string Revision { get; set; } = "main";

    [CommandOption("--token <TOKEN>")]
    [Description("Optional HF access token (env HF_TOKEN also honored).")]
    public string? Token { get; set; }

    public override ValidationResult Validate()
    {
        if (string.IsNullOrWhiteSpace(Repo))
            return ValidationResult.Error("Repo is required.");

        // Also error if it's not a proper url
        if (Repo.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        {
            if (!Uri.TryCreate(Repo, UriKind.Absolute, out var uri) ||
                uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps ||
                uri.Segments.Length < 3)
            {
                return ValidationResult.Error("Repo must be a valid URL or org/name.");
            }
        }

        return ValidationResult.Success();
    }
}
