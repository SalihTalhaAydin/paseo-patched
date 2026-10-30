import { Buffer } from "node:buffer";
import type { Logger } from "pino";
import { v4 as uuidv4 } from "uuid";

import { Pcm16MonoResampler } from "../../agent/pcm16-resampler.js";
import { parsePcmRateFromFormat } from "../../speech/audio.js";
import type {
  SpeechToTextProvider,
  StreamingTranscriptionEvent,
  StreamingTranscriptionSession,
} from "../../speech/speech-provider.js";
import type { TurnDetectionProvider } from "../../speech/turn-detection-provider.js";

const VOICE_FINAL_TRANSCRIPT_TIMEOUT_MS = 10_000;
const MAX_TIMED_OUT_TURNS = 16;

const FILLER_PARTIAL_WORDS = new Set([
  "uh",
  "um",
  "ah",
  "eh",
  "er",
  "hmm",
  "mm",
  "mmm",
  "mhm",
  "huh",
  "uhhuh",
  "uh-huh",
  "oh",
]);

function isFillerOnlyPartial(transcript: string): boolean {
  const tokens = transcript
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s'-]/gu, "")
    .split(/\s+/)
    .filter(Boolean);
  if (tokens.length === 0) {
    return true;
  }
  return tokens.every((token) => FILLER_PARTIAL_WORDS.has(token));
}

type VoiceInputState =
  | { status: "idle" }
  | { status: "listening" }
  | {
      status: "capturing";
      utteranceId: string;
      startedAt: number;
    };

export interface VoiceTurnControllerCallbacks {
  onSpeechStarted(): Promise<void>;
  onSpeechStopped(): Promise<void>;
  onPartialTranscript(input: { segmentId: string; transcript: string }): Promise<void>;
  onFinalTranscript(input: VoiceFinalTranscript): Promise<void>;
  onError(error: Error): void;
}

export interface VoiceTurnController {
  start(): Promise<void>;
  stop(): Promise<void>;
  appendClientChunk(input: { audioBase64: string; format: string }): Promise<void>;
}

interface TranscriptSegmentMeta {
  language?: string;
  avgLogprob?: number;
  isLowConfidence?: boolean;
}

interface VoiceFinalTranscript {
  segmentId: string;
  transcript: string;
  language?: string;
  avgLogprob?: number;
  isLowConfidence?: boolean;
  durationMs: number;
}

interface FinalizingVoiceTurn {
  turnId: string;
  sequence: number;
  startedAt: number;
  committedSegmentIds: string[];
  transcriptsBySegmentId: Map<string, string>;
  finalTranscriptSegmentIds: Set<string>;
  transcriptMetaBySegmentId: Map<string, TranscriptSegmentMeta>;
  timeout: ReturnType<typeof setTimeout>;
  timedOut: boolean;
}

export function createVoiceTurnController(params: {
  logger: Logger;
  turnDetection: TurnDetectionProvider;
  stt: SpeechToTextProvider;
  sttLanguage?: string;
  callbacks: VoiceTurnControllerCallbacks;
}): VoiceTurnController {
  const detector = params.turnDetection.createSession({
    logger: params.logger.child({ component: "turn-detection" }),
  });

  let state: VoiceInputState = { status: "idle" };
  let resampler: Pcm16MonoResampler | null = null;
  let sttSession: StreamingTranscriptionSession | null = null;
  let sttResampler: Pcm16MonoResampler | null = null;
  let inputRate = detector.requiredSampleRate;
  let sttInputRate = 0;
  let queued = Promise.resolve();
  let activeTranscriptSegmentId: string | null = null;
  let partialTranscriptFired = false;
  let reconnectAttemptedForTurn = false;
  const sealedTranscriptSegmentIds = new Set<string>();
  let pendingTurns: FinalizingVoiceTurn[] = [];
  let timedOutTurns: FinalizingVoiceTurn[] = [];
  let nextTurnSequence = 0;
  let lastDeliveredSequence = -1;

  function fail(error: unknown): void {
    params.callbacks.onError(error instanceof Error ? error : new Error(String(error)));
  }

  function firePartialTranscript(segmentId: string, transcript: string): void {
    if (partialTranscriptFired || state.status !== "capturing") {
      return;
    }

    partialTranscriptFired = true;
    void runSerial(async () => {
      await params.callbacks.onPartialTranscript({ segmentId, transcript });
    });
  }

  function clearFinalizingTurns(): void {
    for (const turn of pendingTurns) {
      clearTimeout(turn.timeout);
    }
    pendingTurns = [];
    timedOutTurns = [];
  }

  function getNewestPendingTurn(): FinalizingVoiceTurn | null {
    return pendingTurns[pendingTurns.length - 1] ?? null;
  }

  function getFinalizingTurnForSegment(segmentId: string): FinalizingVoiceTurn | null {
    const committedTurn = pendingTurns.find((turn) => turn.committedSegmentIds.includes(segmentId));
    if (committedTurn) {
      return committedTurn;
    }

    const newestTurn = getNewestPendingTurn();
    if (!newestTurn || newestTurn.committedSegmentIds.length > 0) {
      return null;
    }

    if (activeTranscriptSegmentId && activeTranscriptSegmentId !== segmentId) {
      return null;
    }

    return newestTurn;
  }

  function getOrderedFinalSegmentIds(turn: FinalizingVoiceTurn): string[] {
    if (turn.committedSegmentIds.length === 0) {
      return [...turn.finalTranscriptSegmentIds];
    }

    return turn.committedSegmentIds.filter((segmentId) =>
      turn.finalTranscriptSegmentIds.has(segmentId),
    );
  }

  function assembleFinalTranscript(turn: FinalizingVoiceTurn): VoiceFinalTranscript {
    const orderedFinalSegmentIds = getOrderedFinalSegmentIds(turn);
    const transcript = orderedFinalSegmentIds
      .map((segmentId) => turn.transcriptsBySegmentId.get(segmentId)?.trim() ?? "")
      .filter((segment) => segment.length > 0)
      .join(" ")
      .trim();
    const orderedFinalMeta = orderedFinalSegmentIds
      .map((segmentId) => turn.transcriptMetaBySegmentId.get(segmentId))
      .filter((meta): meta is TranscriptSegmentMeta => Boolean(meta));
    const language = orderedFinalMeta.find((meta) => meta.language)?.language;
    const singleSegmentMeta = orderedFinalMeta.length === 1 ? orderedFinalMeta[0] : null;
    const allLowConfidence =
      orderedFinalMeta.length > 0 &&
      orderedFinalMeta.every((meta) => meta.isLowConfidence === true);

    return {
      segmentId: turn.committedSegmentIds[0] ?? orderedFinalSegmentIds[0] ?? turn.turnId,
      transcript,
      ...(language ? { language } : {}),
      ...(singleSegmentMeta?.avgLogprob !== undefined
        ? { avgLogprob: singleSegmentMeta.avgLogprob }
        : {}),
      ...(allLowConfidence ? { isLowConfidence: true } : {}),
      durationMs: Math.max(0, Date.now() - turn.startedAt),
    };
  }

  function deliverFinalTranscript(
    turn: FinalizingVoiceTurn,
    finalTranscript: VoiceFinalTranscript,
  ): void {
    if (finalTranscript.transcript.length > 0) {
      lastDeliveredSequence = Math.max(lastDeliveredSequence, turn.sequence);
    }

    void runSerial(async () => {
      await params.callbacks.onFinalTranscript(finalTranscript);
    });
  }

  function fireFinalTranscript(turn: FinalizingVoiceTurn, reason: "complete" | "timeout"): void {
    clearTimeout(turn.timeout);

    const finalTranscript = assembleFinalTranscript(turn);
    if (reason === "timeout") {
      params.logger.warn(
        {
          turnId: turn.turnId,
          committedSegments: turn.committedSegmentIds.length,
          receivedFinals: turn.finalTranscriptSegmentIds.size,
          timeoutMs: VOICE_FINAL_TRANSCRIPT_TIMEOUT_MS,
          transcriptLength: finalTranscript.transcript.length,
        },
        "voice_turn.final_transcript_timeout",
      );
      if (finalTranscript.transcript.length === 0 && turn.committedSegmentIds.length > 0) {
        timedOutTurns.push(turn);
        if (timedOutTurns.length > MAX_TIMED_OUT_TURNS) {
          timedOutTurns.shift();
        }
      }
    }

    deliverFinalTranscript(turn, finalTranscript);
  }

  function isTurnComplete(turn: FinalizingVoiceTurn): boolean {
    return (
      turn.committedSegmentIds.length > 0 &&
      turn.committedSegmentIds.every((segmentId) => turn.finalTranscriptSegmentIds.has(segmentId))
    );
  }

  function flushResolvedTurns(): void {
    let head = pendingTurns[0];
    while (head && (head.timedOut || isTurnComplete(head))) {
      pendingTurns.shift();
      fireFinalTranscript(head, isTurnComplete(head) ? "complete" : "timeout");
      head = pendingTurns[0];
    }
  }

  function recordFinalTranscript(
    turn: FinalizingVoiceTurn,
    event: StreamingTranscriptionEvent,
  ): void {
    turn.transcriptsBySegmentId.set(event.segmentId, event.transcript);
    turn.finalTranscriptSegmentIds.add(event.segmentId);
    turn.transcriptMetaBySegmentId.set(event.segmentId, {
      ...(event.language ? { language: event.language } : {}),
      ...(event.avgLogprob !== undefined ? { avgLogprob: event.avgLogprob } : {}),
      ...(event.isLowConfidence !== undefined ? { isLowConfidence: event.isLowConfidence } : {}),
    });
  }

  function handleLateFinalTranscript(event: StreamingTranscriptionEvent): void {
    const index = timedOutTurns.findIndex((turn) =>
      turn.committedSegmentIds.includes(event.segmentId),
    );
    const turn = timedOutTurns[index];
    const transcriptLength = event.transcript.trim().length;
    if (!turn || transcriptLength === 0) {
      return;
    }

    timedOutTurns.splice(index, 1);
    if (lastDeliveredSequence > turn.sequence) {
      params.logger.warn(
        { turnId: turn.turnId, transcriptLength },
        "voice_turn.late_transcript_dropped",
      );
      return;
    }

    recordFinalTranscript(turn, event);
    params.logger.info(
      { turnId: turn.turnId, transcriptLength },
      "voice_turn.late_transcript_delivered",
    );
    deliverFinalTranscript(turn, assembleFinalTranscript(turn));
  }

  async function reconnectSttSession(): Promise<void> {
    const previousSession = sttSession;
    sttSession = null;
    sttResampler = null;
    sttInputRate = 0;
    previousSession?.close();

    try {
      const nextSession = createSttSession();
      await nextSession.connect();
      sttSession = nextSession;
      params.logger.info("voice_turn.stt_reconnected");
    } catch (error) {
      fail(error);
      params.logger.warn({ err: error }, "voice_turn.stt_reconnect_failed");
    }
  }

  function handleSttError(error: unknown): void {
    fail(error);
    params.logger.warn({ err: error }, "voice_turn.stt_error");
    if (reconnectAttemptedForTurn) {
      sttSession?.close();
      sttSession = null;
      return;
    }

    reconnectAttemptedForTurn = true;
    void runSerial(reconnectSttSession);
  }

  function handleFinalSttTranscript(event: StreamingTranscriptionEvent): void {
    const turn = getFinalizingTurnForSegment(event.segmentId);
    if (!turn) {
      handleLateFinalTranscript(event);
      return;
    }

    recordFinalTranscript(turn, event);
    flushResolvedTurns();
  }

  function handlePartialSttTranscript(event: StreamingTranscriptionEvent): void {
    if (state.status !== "capturing" || partialTranscriptFired) {
      return;
    }

    if (sealedTranscriptSegmentIds.has(event.segmentId)) {
      return;
    }

    if (activeTranscriptSegmentId && event.segmentId !== activeTranscriptSegmentId) {
      return;
    }

    const transcript = event.transcript.trim();
    if (!transcript) {
      return;
    }

    activeTranscriptSegmentId = event.segmentId;

    if (isFillerOnlyPartial(transcript)) {
      return;
    }

    firePartialTranscript(event.segmentId, transcript);
  }

  function handleSttTranscript(event: StreamingTranscriptionEvent): void {
    if (event.isFinal) {
      handleFinalSttTranscript(event);
      return;
    }

    handlePartialSttTranscript(event);
  }

  function createSttSession(): StreamingTranscriptionSession {
    const session = params.stt.createSession({
      logger: params.logger.child({ component: "stt" }),
      language: params.sttLanguage ?? "en",
    });
    session.on("transcript", handleSttTranscript);
    session.on("committed", ({ segmentId }) => {
      sealedTranscriptSegmentIds.add(segmentId);
      if (state.status === "capturing" && !activeTranscriptSegmentId) {
        activeTranscriptSegmentId = segmentId;
      }
      const turn = getNewestPendingTurn();
      if (turn && !turn.committedSegmentIds.includes(segmentId)) {
        turn.committedSegmentIds.push(segmentId);
        flushResolvedTurns();
      }
    });
    session.on("error", handleSttError);
    return session;
  }

  function runSerial(task: () => Promise<void>): Promise<void> {
    queued = queued.then(task).catch((error) => {
      fail(error);
    });
    return queued;
  }

  function updateDetectorResampler(parsedInputRate: number): void {
    if (parsedInputRate === inputRate) {
      return;
    }

    inputRate = parsedInputRate;
    resampler =
      inputRate === detector.requiredSampleRate
        ? null
        : new Pcm16MonoResampler({
            inputRate,
            outputRate: detector.requiredSampleRate,
          });
  }

  function updateSttResampler(
    session: StreamingTranscriptionSession,
    parsedInputRate: number,
  ): void {
    if (parsedInputRate === sttInputRate) {
      return;
    }

    sttInputRate = parsedInputRate;
    sttResampler =
      sttInputRate === session.requiredSampleRate
        ? null
        : new Pcm16MonoResampler({
            inputRate: sttInputRate,
            outputRate: session.requiredSampleRate,
          });
  }

  function pcmForDetector(pcm16: Buffer): Buffer {
    return resampler === null ? pcm16 : resampler.processChunk(pcm16);
  }

  function pcmForStt(
    session: StreamingTranscriptionSession,
    pcm16: Buffer,
    detectorPcm16: Buffer,
  ): Buffer {
    if (session.requiredSampleRate === detector.requiredSampleRate) {
      return detectorPcm16;
    }

    if (sttResampler === null) {
      return pcm16;
    }

    return sttResampler.processChunk(pcm16);
  }

  async function handleSpeechStarted(): Promise<void> {
    if (state.status === "capturing") {
      return;
    }

    await params.callbacks.onSpeechStarted();

    const startedAt = Date.now();
    activeTranscriptSegmentId = null;
    partialTranscriptFired = false;
    reconnectAttemptedForTurn = false;
    state = {
      status: "capturing",
      utteranceId: uuidv4(),
      startedAt,
    };
    params.logger.info(
      {
        utteranceId: state.utteranceId,
      },
      "voice_turn.speech_started",
    );
  }

  async function handleSpeechStopped(): Promise<void> {
    if (state.status !== "capturing") {
      return;
    }

    const turnId = state.utteranceId;
    const startedAt = state.startedAt;
    const endedAt = Date.now();

    state = { status: "listening" };

    const finalizingTurn: FinalizingVoiceTurn = {
      turnId,
      sequence: nextTurnSequence++,
      startedAt,
      committedSegmentIds: [],
      transcriptsBySegmentId: new Map(),
      finalTranscriptSegmentIds: new Set(),
      transcriptMetaBySegmentId: new Map(),
      timeout: setTimeout(() => {
        finalizingTurn.timedOut = true;
        flushResolvedTurns();
      }, VOICE_FINAL_TRANSCRIPT_TIMEOUT_MS),
      timedOut: false,
    };
    pendingTurns.push(finalizingTurn);

    detector.reset();
    try {
      sttSession?.commit();
    } catch (error) {
      handleSttError(error);
    }

    await params.callbacks.onSpeechStopped();

    params.logger.info(
      {
        utteranceAgeMs: Math.max(0, endedAt - startedAt),
      },
      "voice_turn.speech_stopped",
    );
  }

  detector.on("speech_started", () => {
    void runSerial(handleSpeechStarted);
  });
  detector.on("speech_stopped", () => {
    void runSerial(handleSpeechStopped);
  });
  detector.on("error", fail);

  return {
    async start(): Promise<void> {
      sttSession = createSttSession();
      await sttSession.connect();
      await detector.connect();
      state = { status: "listening" };
    },

    async stop(): Promise<void> {
      await runSerial(async () => {
        clearFinalizingTurns();
        detector.close();
        sttSession?.close();
        resampler?.reset();
        sttResampler?.reset();
        resampler = null;
        sttResampler = null;
        sttSession = null;
        state = { status: "idle" };
      });
    },

    async appendClientChunk(input): Promise<void> {
      await runSerial(async () => {
        if (state.status === "idle") {
          return;
        }

        const pcm16 = Buffer.from(input.audioBase64, "base64");
        if (pcm16.length === 0) {
          return;
        }

        const parsedInputRate =
          parsePcmRateFromFormat(input.format, detector.requiredSampleRate) ??
          detector.requiredSampleRate;

        updateDetectorResampler(parsedInputRate);
        const currentSttSession = sttSession;
        if (currentSttSession) {
          updateSttResampler(currentSttSession, parsedInputRate);
        }

        const detectorPcm16 = pcmForDetector(pcm16);
        const sttPcm16 = currentSttSession
          ? pcmForStt(currentSttSession, pcm16, detectorPcm16)
          : null;
        if (detectorPcm16.length === 0 && (!sttPcm16 || sttPcm16.length === 0)) {
          return;
        }

        if (detectorPcm16.length > 0) {
          detector.appendPcm16(detectorPcm16);
        }

        if (sttPcm16 && sttPcm16.length > 0) {
          try {
            currentSttSession?.appendPcm16(sttPcm16);
          } catch (error) {
            handleSttError(error);
          }
        }
      });
    },
  };
}
