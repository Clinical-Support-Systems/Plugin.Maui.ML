using CommunityToolkit.Maui;
using Microsoft.Extensions.Logging;
using Plugin.Maui.ML;
using Plugin.Maui.ML.Configuration;
using Syncfusion.Maui.Toolkit.Hosting;

namespace MauiSample
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCommunityToolkit()
                .ConfigureSyncfusionToolkit()
                // ReSharper disable once UnusedParameter.Local
#pragma warning disable RCS1163
                .ConfigureMauiHandlers(handlers =>
#pragma warning restore RCS1163
                {
#if IOS || MACCATALYST
                    handlers.AddHandler<Microsoft.Maui.Controls.CollectionView, Microsoft.Maui.Controls.Handlers.Items2.CollectionViewHandler2>();
#endif
                })
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                    fonts.AddFont("SegoeUI-Semibold.ttf", "SegoeSemibold");
                    fonts.AddFont("FluentSystemIcons-Regular.ttf", FluentUI.FontFamily);
                });

            // Register ML services
            builder.Services.AddMauiML(config =>
            {
                config.UseTransientService = false;
                config.EnablePerformanceLogging = true;
            });

            // Register config provider (looks for BiomedicalNER.config.json at app base directory / platform asset copy)
            builder.Services.AddSingleton<INlpModelConfigProvider>(_ => new FileSystemNlpModelConfigProvider());

#if DEBUG
            builder.Logging.AddDebug();
            builder.Services.AddLogging(configure => configure.AddDebug());
#endif

            builder.Services.AddSingleton<IMedicalNlpService, MedicalNlpService>();

            builder.Services.AddSingleton<ModalErrorHandler>();
            builder.Services.AddSingleton<MainPageModel>();

            return builder.Build();
        }
    }
}
