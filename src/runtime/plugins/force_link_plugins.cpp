// Force the linker to include all self-registering plugin object files.
// Without this, the linker may strip plugin .o files from the static library
// because nothing in the main code directly references their symbols.
// Each plugin defines a volatile int that we reference here.

namespace trtf {

#if TRTF_HAS_TRT

// Declared in each plugin .cpp
extern volatile int kForceLink_DecoderPlugin;
extern volatile int kForceLink_SsmPlugin;
extern volatile int kForceLink_RwkvPlugin;
extern volatile int kForceLink_HybridPlugin;
extern volatile int kForceLink_EncoderPlugin;
extern volatile int kForceLink_SegmentationPlugin;
extern volatile int kForceLink_ObjectDetectionPlugin;
extern volatile int kForceLink_VLPlugin;
extern volatile int kForceLink_WhisperPlugin;
extern volatile int kForceLink_BarkPlugin;
extern volatile int kForceLink_MagpiePlugin;
extern volatile int kForceLink_SpeechPlugin;
extern volatile int kForceLink_OmniPlugin;
extern volatile int kForceLink_FluxPlugin;
extern volatile int kForceLink_WanPlugin;
extern volatile int kForceLink_ZImagePlugin;
extern volatile int kForceLink_T5Plugin;
extern volatile int kForceLink_MarianPlugin;
extern volatile int kForceLink_Seq2SeqPlugin;
extern volatile int kForceLink_PixArtPlugin;
extern volatile int kForceLink_PixArtTorchTrtPlugin;

// Referenced from pipeline_registry.cpp to ensure this TU is linked.
volatile int* force_link_all_plugins() {
    static volatile int* kPluginAnchors[] = {
        &kForceLink_DecoderPlugin,
        &kForceLink_SsmPlugin,
        &kForceLink_RwkvPlugin,
        &kForceLink_HybridPlugin,
        &kForceLink_EncoderPlugin,
        &kForceLink_SegmentationPlugin,
        &kForceLink_ObjectDetectionPlugin,
        &kForceLink_VLPlugin,
        &kForceLink_WhisperPlugin,
        &kForceLink_BarkPlugin,
        &kForceLink_MagpiePlugin,
        &kForceLink_SpeechPlugin,
        &kForceLink_OmniPlugin,
        &kForceLink_FluxPlugin,
        &kForceLink_WanPlugin,
        &kForceLink_ZImagePlugin,
        &kForceLink_T5Plugin,
        &kForceLink_MarianPlugin,
        &kForceLink_Seq2SeqPlugin,
        &kForceLink_PixArtPlugin,
        &kForceLink_PixArtTorchTrtPlugin,
    };
    return kPluginAnchors[0];
}

#endif // TRTF_HAS_TRT

} // namespace trtf
