import pino from "pino";
import { afterEach, describe, expect, test, vi } from "vitest";

const { openAiConstructorOptionsMock, speechCreateMock } = vi.hoisted(() => ({
  openAiConstructorOptionsMock: vi.fn(),
  speechCreateMock: vi.fn(),
}));

vi.mock("openai", () => ({
  OpenAI: vi.fn(function OpenAI(options: unknown) {
    openAiConstructorOptionsMock(options);
    return {
      audio: {
        speech: {
          create: speechCreateMock,
        },
      },
    };
  }),
}));

import { OpenAITTS } from "./tts.js";

describe("OpenAITTS", () => {
  afterEach(() => {
    openAiConstructorOptionsMock.mockReset();
    speechCreateMock.mockReset();
  });

  test("passes configured baseUrl to the OpenAI client", () => {
    const provider = new OpenAITTS(
      { apiKey: "sk-test", baseUrl: "https://speech.example.com/v1" },
      pino({ level: "silent" }),
    );

    expect(provider.getConfig().baseUrl).toBe("https://speech.example.com/v1");
    expect(openAiConstructorOptionsMock).toHaveBeenCalledWith({
      apiKey: "sk-test",
      baseURL: "https://speech.example.com/v1",
    });
  });

  test("strips a WAV container when the server ignores the pcm response format", async () => {
    const samples = Buffer.from([1, 0, 2, 0, 3, 0, 4, 0]);
    const header = Buffer.alloc(44);
    header.write("RIFF", 0, "ascii");
    header.writeUInt32LE(36 + samples.length, 4);
    header.write("WAVE", 8, "ascii");
    header.write("fmt ", 12, "ascii");
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(1, 22);
    header.writeUInt32LE(24000, 24);
    header.writeUInt32LE(48000, 28);
    header.writeUInt16LE(2, 32);
    header.writeUInt16LE(16, 34);
    header.write("data", 36, "ascii");
    header.writeUInt32LE(samples.length, 40);
    const wav = Buffer.concat([header, samples]);
    speechCreateMock.mockResolvedValue({
      arrayBuffer: async () => wav.buffer.slice(wav.byteOffset, wav.byteOffset + wav.byteLength),
    });
    const provider = new OpenAITTS({ apiKey: "sk-test" }, pino({ level: "silent" }));

    const result = await provider.synthesizeSpeech("Hello");
    const chunks: Buffer[] = [];
    for await (const chunk of result.stream) chunks.push(Buffer.from(chunk));

    expect(result.format).toBe("pcm");
    expect(Buffer.concat(chunks)).toEqual(samples);
  });

  test("passes raw pcm through unchanged", async () => {
    const samples = Buffer.from([5, 0, 6, 0]);
    speechCreateMock.mockResolvedValue({
      arrayBuffer: async () =>
        samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength),
    });
    const provider = new OpenAITTS({ apiKey: "sk-test" }, pino({ level: "silent" }));

    const result = await provider.synthesizeSpeech("Hello");
    const chunks: Buffer[] = [];
    for await (const chunk of result.stream) chunks.push(Buffer.from(chunk));

    expect(Buffer.concat(chunks)).toEqual(samples);
  });
});
