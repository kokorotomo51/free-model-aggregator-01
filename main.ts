// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief, probes every provider's catalog route, builds one catalog, and
// then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the auth header it got),
//      POST /v1/chat/completions and POST /v1/messages, each as JSON or server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
//
// Routing is where the score lives. For each request the server ranks every free model that meets
// the request's requirements by the quality table's rating_milli, best first, and takes the first
// provider that answers. A provider that refuses (402, 429 or any other failure) is retried twice
// on the same model — the env's failure schedule refuses single calls, so a retry usually lands —
// and only after three refusals is the provider left alone for the rest of the trial and the
// request failed over to the next best model. One successful provider call answers a request:
// a second success would break the trial's one-provider check, so the loop stops at the first.
//
// A JSON-mode request (an OpenAI client asking for response_format) never goes to an
// Anthropic-wire provider: that wire cannot carry JSON mode and its echo would say so. The
// catalog, the model list and every answer are pure functions of the brief, the requests and the
// providers' answers: no clock and no unseeded randomness, so a replay leaves the same export.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; run
// outside a trial, with no brief and no env, it prints {"providers":[]}.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelIsFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Catalog,
  type DirectoryProvider,
  type Json,
  type ListedModel,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerBody,
  anthropicAnswerStream,
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  clientToGemini,
  openAIAnswerBody,
  openAIAnswerStream,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** How many attempts one provider gets for one request before it is left alone. */
const ATTEMPTS_PER_PROVIDER = 3;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider so far, refusals included. */
const callsMade = new Map<string, number>();
/** Providers that refused three times in a row: never called again this trial. */
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models the catalog holds, in brief order: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: DirectoryProvider;
  model: ListedModel;
  rating: number;
}

/**
 * Every free model that can serve a request, best first. The order is the quality table's rating,
 * then the provider with the fewest calls so far (spread the load over the providers that are
 * still answering), then provider and model id: fully determined by the trial so far, so a replay
 * ranks the same way. A JSON-mode request skips Anthropic-wire providers: that wire cannot carry
 * JSON mode and its echo would say the request never asked for it.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (deadProviders.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null) continue;
      if (!modelIsFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row.rating_milli });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const callsA = callsMade.get(a.provider.id) ?? 0;
    const callsB = callsMade.get(b.provider.id) ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === clientApi) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether this client request asks for JSON mode (an OpenAI response_format). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the provider's wire, its auth, the model it names. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstreamBody: Json =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id }
        : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody: Json =
      clientApi === 'anthropic'
        ? { ...body, model: candidate.model.id }
        : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const encoded = encodeURIComponent(candidate.model.id);
  const path = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
  return { url: `${base}/v1beta/models/${encoded}${path}`, headers, body: clientToGemini(body) };
}

interface ProviderResult {
  ok: boolean;
  status: number;
  text: string;
}

/** One inference call to a provider: its status and body text, or a refusal on a transport failure. */
async function callProvider(call: UpstreamCall): Promise<ProviderResult> {
  try {
    const res = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
    return { ok: res.status >= 200 && res.status < 300, status: res.status, text: await res.text() };
  } catch (err) {
    return { ok: false, status: 0, text: `the provider could not be reached: ${(err as Error).message}` };
  }
}

/** Parse a provider's answer text into one neutral answer, by the provider's wire. */
function parseAnswer(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): UpstreamAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

/**
 * Serve one client request: try the candidates in order, retrying one provider up to three times
 * before leaving it alone and failing over. Returns the provider's answer and which model gave it,
 * or null when nothing answered.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<{ answer: UpstreamAnswer; provider: DirectoryProvider; model: string; rawText: string } | null> {
  const requires = requirementsFor(clientApi, body);
  const jsonMode = clientApi === 'openai' && jsonModeOf(body);
  const candidates = candidatesFor(requires, jsonMode);
  for (const candidate of candidates) {
    const providerId = candidate.provider.id;
    let refused = 0;
    for (let attempt = 0; attempt < ATTEMPTS_PER_PROVIDER; attempt += 1) {
      callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
      const result = await callProvider(upstreamCall(candidate, clientApi, body, stream));
      if (result.ok) {
        const answer = parseAnswer(candidate.provider.wire, result.text, stream);
        return {
          answer: answer ?? { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] },
          provider: candidate.provider,
          model: candidate.model.id,
          rawText: result.text,
        };
      }
      refused += 1;
      log(`${providerId}/${candidate.model.id} refused (attempt ${attempt + 1}, HTTP ${result.status})`);
    }
    if (refused >= ATTEMPTS_PER_PROVIDER) deadProviders.add(providerId);
  }
  return null;
}

/** Answer the client in its own wire, from the provider's answer. */
function answerClient(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  providerWire: 'openai' | 'anthropic' | 'gemini',
  stream: boolean,
  served: { answer: UpstreamAnswer; model: string; rawText: string },
): void {
  if (clientApi === providerWire) {
    // Same wire: the provider's bytes are already the client's shape. Forward them verbatim.
    const contentType = stream ? 'text/event-stream; charset=utf-8' : 'application/json';
    sendText(res, 200, contentType, served.rawText);
    return;
  }
  if (clientApi === 'openai') {
    if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', openAIAnswerStream(served.answer, served.model));
    else sendJson(res, 200, openAIAnswerBody(served.answer, served.model));
    return;
  }
  if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicAnswerStream(served.answer, served.model));
  else sendJson(res, 200, anthropicAnswerBody(served.answer, served.model));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body, stream);
  if (served === null) return sendError(res, 'openai', 503, 'server_error', 'no free model on the directory meets this request');
  answerClient(res, 'openai', served.provider.wire, stream, served);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body, stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  answerClient(res, 'anthropic', served.provider.wire, stream, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
