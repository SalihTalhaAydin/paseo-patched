import type pino from "pino";
import { OpenAI } from "openai";
import { Readable } from "node:stream";
import type { SpeechStreamResult, TextToSpeechProvider } from "../../speech-provider.js";

export type { SpeechStreamResult };

export interface TTSConfig {
  apiKey: string;
  baseUrl?: string;
  model?: "tts-1" | "tts-1-hd" | (string & {});
  voice?: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" | (string & {});
  responseFormat?: "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm";
}

export class OpenAITTS implements TextToSpeechProvider {
  private readonly openaiClient: OpenAI;
  private readonly config: TTSConfig;
  private readonly logger: pino.Logger;

  constructor(ttsConfig: TTSConfig, parentLogger: pino.Logger) {
    this.config = {
      model: "tts-1",
      voice: "alloy",
      responseFormat: "pcm",
      ...ttsConfig,
    };
    this.logger = parentLogger.child({ module: "agent", provider: "openai", component: "tts" });
    this.openaiClient = new OpenAI({
      apiKey: ttsConfig.apiKey,
      ...(ttsConfig.baseUrl ? { baseURL: ttsConfig.baseUrl } : {}),
    });

    this.logger.info(
      { voice: this.config.voice, model: this.config.model, format: this.config.responseFormat },
      "TTS (OpenAI) initialized",
    );
  }

  public getConfig(): TTSConfig {
    return this.config;
  }

  public async synthesizeSpeech(text: string): Promise<SpeechStreamResult> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot synthesize empty text");
    }

    const startTime = Date.now();

    try {
      this.logger.debug(
        { textLength: text.length, preview: text.substring(0, 50) },
        "Synthesizing speech",
      );

      const response = await this.openaiClient.audio.speech.create({
        model: this.config.model!,
        voice: this.config.voice!,
        input: text,
        response_format: this.config.responseFormat as
          | "mp3"
          | "opus"
          | "aac"
          | "flac"
          | "wav"
          | "pcm",
      });

      // Prefetched speech can wait through playback of an earlier sentence. Consume
      // the response here so fetch cannot collect it and cancel its unread body.
      let audio: Buffer = Buffer.from(await response.arrayBuffer());
      if (this.config.responseFormat === "pcm") {
        audio = this.stripWavContainer(audio);
      }
      const audioStream = Readable.from([audio]);

      const duration = Date.now() - startTime;
      this.logger.debug({ duration }, "Speech synthesis stream ready");

      return {
        stream: audioStream,
        format: this.config.responseFormat || "mp3",
      };
    } catch (error) {
      this.logger.error({ err: error }, "Speech synthesis error");
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`TTS synthesis failed: ${message}`, { cause: error });
    }
  }

  // Some OpenAI-compatible servers ignore `response_format: "pcm"` and return a WAV file.
  // Played as raw PCM, its header is an audible pop at the start of every sentence.
  private stripWavContainer(audio: Buffer): Buffer {
    if (
      audio.length < 12 ||
      audio.toString("ascii", 0, 4) !== "RIFF" ||
      audio.toString("ascii", 8, 12) !== "WAVE"
    ) {
      return audio;
    }
    let offset = 12;
    while (offset + 8 <= audio.length) {
      const chunkId = audio.toString("ascii", offset, offset + 4);
      const chunkSize = audio.readUInt32LE(offset + 4);
      const body = offset + 8;
      if (chunkId === "fmt " && body + 16 <= audio.length) {
        const sampleRate = audio.readUInt32LE(body + 4);
        const channels = audio.readUInt16LE(body + 2);
        const bitsPerSample = audio.readUInt16LE(body + 14);
        if (sampleRate !== 24000 || channels !== 1 || bitsPerSample !== 16) {
          this.logger.warn(
            { sampleRate, channels, bitsPerSample },
            "TTS server returned WAV that does not match 24kHz mono 16-bit PCM",
          );
        }
      }
      if (chunkId === "data") {
        return audio.subarray(body, Math.min(body + chunkSize, audio.length));
      }
      offset = body + chunkSize + (chunkSize % 2);
    }
    return audio;
  }
}
