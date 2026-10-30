import pino from "pino";
import { afterEach, describe, expect, test, vi } from "vitest";

const { openAiConstructorOptionsMock, transcriptionsCreateMock } = vi.hoisted(() => ({
  openAiConstructorOptionsMock: vi.fn(),
  transcriptionsCreateMock: vi.fn(),
}));

vi.mock("openai", () => ({
  OpenAI: vi.fn(function OpenAI(options: unknown) {
    openAiConstructorOptionsMock(options);
    return {
      audio: {
        transcriptions: {
          create: transcriptionsCreateMock,
        },
      },
    };
  }),
}));

import { OpenAISTT } from "./stt.js";

describe("OpenAISTT", () => {
  afterEach(() => {
    openAiConstructorOptionsMock.mockReset();
    transcriptionsCreateMock.mockReset();
  });

  test("passes configured baseUrl to the OpenAI client", () => {
    const provider = new OpenAISTT(
      { apiKey: "sk-test", baseUrl: "https://speech.example.com/v1" },
      pino({ level: "silent" }),
    );

    expect(provider.id).toBe("openai");
    expect(openAiConstructorOptionsMock).toHaveBeenCalledWith({
      apiKey: "sk-test",
      baseURL: "https://speech.example.com/v1",
    });
  });

  test("passes transcription prompt to OpenAI REST STT", async () => {
    mockTranscriptionResponse("hello");

    await expect(
      transcribe({ language: "en", prompt: "Only transcribe the speaker." }),
    ).resolves.toBe("hello");
    expect(transcriptionsCreateMock).toHaveBeenCalledWith(
      expect.objectContaining({
        language: "en",
        model: "gpt-4o-transcribe",
        prompt: "Only transcribe the speaker.",
        response_format: "json",
      }),
      expect.anything(),
    );
  });

  test("omits the language hint when language is auto", async () => {
    mockTranscriptionResponse("hello");

    await expect(transcribe({ language: "auto" })).resolves.toBe("hello");
    expect(transcriptionsCreateMock).toHaveBeenCalledTimes(1);
    expect(transcriptionsCreateMock.mock.calls[0][0]).not.toHaveProperty("language");
  });

  test("bounds each transcription request with a timeout and one retry", async () => {
    mockTranscriptionResponse("hello");

    await transcribe({ language: "en" });

    expect(transcriptionsCreateMock).toHaveBeenCalledWith(expect.anything(), {
      timeout: 8000,
      maxRetries: 1,
    });
  });

  test("keeps audio appended while a commit is in flight for the next commit", async () => {
    const pending = deferred<void>();
    const sentWavs: Buffer[] = [];
    transcriptionsCreateMock.mockImplementation(
      async (request: { file: NodeJS.ReadableStream }) => {
        sentWavs.push(await readStream(request.file));
        if (sentWavs.length === 1) {
          await pending.promise;
        }
        return { text: `request ${sentWavs.length}` };
      },
    );
    const session = createSession();
    const transcripts: string[] = [];
    session.on("transcript", (event) => {
      transcripts.push(event.transcript);
    });
    await session.connect();

    session.appendPcm16(Buffer.from([1, 0, 1, 0]));
    session.commit();
    await vi.waitFor(() => expect(sentWavs).toHaveLength(1));

    session.appendPcm16(Buffer.from([2, 0, 2, 0, 2, 0]));
    session.commit();
    await vi.waitFor(() => expect(sentWavs).toHaveLength(2));
    pending.resolve();
    await vi.waitFor(() => expect(transcripts).toHaveLength(2));

    expect(sentWavs[0].subarray(44)).toEqual(Buffer.from([1, 0, 1, 0]));
    expect(sentWavs[1].subarray(44)).toEqual(Buffer.from([2, 0, 2, 0, 2, 0]));
  });

  test("gives each commit its own segment id", async () => {
    const pending = deferred<void>();
    transcriptionsCreateMock.mockImplementation(
      async (request: { file: NodeJS.ReadableStream }) => {
        await readStream(request.file);
        await pending.promise;
        return { text: "hello" };
      },
    );
    const session = createSession();
    const committed: Array<{ segmentId: string; previousSegmentId: string | null }> = [];
    const transcripts: string[] = [];
    session.on("committed", (event) => {
      committed.push(event);
    });
    session.on("transcript", (event) => {
      transcripts.push(event.transcript);
    });
    await session.connect();

    session.appendPcm16(Buffer.from([1, 0]));
    session.commit();
    session.appendPcm16(Buffer.from([2, 0]));
    session.commit();
    pending.resolve();
    await vi.waitFor(() => expect(transcripts).toHaveLength(2));

    expect(committed).toHaveLength(2);
    expect(committed[1].segmentId).not.toBe(committed[0].segmentId);
    expect(committed[1].previousSegmentId).toBe(committed[0].segmentId);
  });
});

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

async function readStream(stream: NodeJS.ReadableStream): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

function createSession() {
  const provider = new OpenAISTT(
    { apiKey: "sk-test", model: "gpt-4o-transcribe" },
    pino({ level: "silent" }),
  );
  return provider.createSession({ logger: pino({ level: "silent" }), language: "en" });
}

function mockTranscriptionResponse(text: string) {
  transcriptionsCreateMock.mockImplementation(async (request: { file: NodeJS.ReadableStream }) => {
    await new Promise<void>((resolve, reject) => {
      request.file.once("error", reject);
      request.file.once("end", resolve);
      request.file.resume();
    });
    return { text };
  });
}

async function transcribe(options: { language: string; prompt?: string }): Promise<string> {
  const provider = new OpenAISTT(
    { apiKey: "sk-test", model: "gpt-4o-transcribe" },
    pino({ level: "silent" }),
  );
  const session = provider.createSession({ logger: pino({ level: "silent" }), ...options });
  const transcript = new Promise<string>((resolve, reject) => {
    session.on("transcript", (event) => {
      if (event.isFinal) {
        resolve(event.transcript);
      }
    });
    session.on("error", (error) => {
      reject(error instanceof Error ? error : new Error(String(error)));
    });
  });

  await session.connect();
  session.appendPcm16(Buffer.from([0, 0, 0, 0]));
  session.commit();
  return transcript;
}
