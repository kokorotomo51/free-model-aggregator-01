// The client wires and the three upstream wires.
//
// The trial's client calls this entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or server-sent events. Each request is
// answered by one provider, on that provider's own wire, and the answer is returned in the wire
// the client used:
//
//   - an OpenAI-wire provider takes the request almost as the client sent it (the model id
//     replaced), and answers it in the same shape;
//   - an Anthropic-wire provider takes a Messages request and answers one;
//   - a Gemini-wire provider takes a generateContent request (the model in the URL path) and
//     answers candidates.
//
// The provider echoes what it parsed — system prompt, user text, images, tools, stop sequences,
// max_tokens, JSON mode — and the client compares that echo with what it sent, so a field the
// translation rewrites or drops loses the request. Every translator below carries the same
// fields through, in the shape the provider's wire reads them.
//
// A response that stays on its own wire is forwarded verbatim (the provider's bytes are already
// the client's shape). A response that crosses a wire is parsed into one answer — text, usage,
// stop reason, tool calls — and rebuilt in the client's shape. Everything here is a pure
// function of its arguments: no clock and no randomness, so the same body and the same provider
// answer give the same bytes on a replay. The model listings use fixed timestamps for the same
// reason.

import { isRecord, type Json } from './providers.ts';

/** One provider answer, in wire-neutral form. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string | null;
  toolCalls: { id: string; name: string; arguments: string }[];
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A finite number, or null when the value is not one. */
function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function strOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of a string or of a block array's text blocks, joined without a separator. */
function textJoined(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** The frames of an SSE body, in order. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

// ---------------------------------------------------------------------------
// Errors and model lists
// ---------------------------------------------------------------------------

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** GET /v1/models on an OpenAI-shaped request: the free models this entry serves. */
export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: the free models this entry serves. */
export function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

// ---------------------------------------------------------------------------
// Stop reasons: the provider's stop, in the client's wire's words (same family).
// ---------------------------------------------------------------------------

function openAIStop(stop: string | null): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

function anthropicStop(stop: string | null): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translation: client body -> upstream body, per wire.
// ---------------------------------------------------------------------------

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic image block as an OpenAI image_url part; null for anything else. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: strOf(block.id),
          type: 'function',
          function: { name: strOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    const toolMessages: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        toolMessages.push({
          role: 'tool',
          tool_call_id: strOf(block.tool_use_id),
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else if (toolMessages.length > 0) messages.push({ role: 'user', content: '' });
    for (const toolMessage of toolMessages) messages.push(toolMessage);
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: strOf(raw.description),
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI image_url part as an Anthropic image block; null for anything else. */
function anthropicPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the system
 * message becomes the system prompt, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap, temperature, top_p
 * and stop sequences pass through. JSON mode has no Anthropic spelling: never route a JSON-mode
 * request to an Anthropic-wire provider.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'user') {
      const content = raw.content;
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = anthropicPartOf(part);
        if (block !== null) blocks.push(block);
      }
      messages.push({ role: 'user', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({
          type: 'tool_use',
          id: strOf(call.id) !== '' ? strOf(call.id) : 'toolu_unknown',
          name: strOf(call.function.name),
          input,
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: strOf(raw.tool_call_id),
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: strOf(raw.function.description),
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The output cap of a request body, from either wire's spelling. */
function maxTokensOf(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/** The stop sequences of a request body, from either wire's spelling. */
function stopsOf(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/** One image URL as a Gemini inlineData or fileData part. */
function geminiImagePart(url: string): Json | null {
  const data = /^data:([^;]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

/** The system text of a body, from either wire's spelling. */
function systemTextOf(body: Json): string {
  const direct = textOf(body.system);
  if (direct !== '') return direct;
  const parts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The function declarations of a body, from either wire's tool spelling. */
function geminiDeclarationsOf(body: Json): Json[] {
  const out: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      if (isRecord(raw.function) && typeof raw.function.name === 'string') {
        out.push({
          name: raw.function.name,
          description: strOf(raw.function.description),
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      } else if (typeof raw.name === 'string') {
        out.push({
          name: raw.name,
          description: strOf(raw.description),
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  return out;
}

/**
 * A client request (either wire) as a Gemini generateContent body. The model goes in the URL path,
 * not the body. The system prompt becomes systemInstruction, messages become contents (assistant
 * turns as model turns, tool results as functionResponse parts), tools become functionDeclarations,
 * and the output cap, stop sequences and JSON mode become the generation config.
 */
export function clientToGemini(body: Json): Json {
  const out: Json = {};
  const system = systemTextOf(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };

  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'user') {
      const content = raw.content;
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image' && isRecord(part.source)) {
            const src = part.source;
            if (src.type === 'base64' && typeof src.data === 'string') {
              parts.push({ inlineData: { mimeType: strOf(src.media_type) !== '' ? strOf(src.media_type) : 'image/png', data: src.data } });
            } else if (src.type === 'url' && typeof src.url === 'string') {
              const image = geminiImagePart(src.url);
              if (image !== null) parts.push(image);
            }
          } else if (part.type === 'tool_result') {
            parts.push({
              functionResponse: { name: strOf(part.tool_use_id), response: { content: toolResultText(part.content) } },
            });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const image = geminiImagePart(part.image_url.url);
            if (image !== null) parts.push(image);
          }
        }
      }
      contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: strOf(call.function.name), args } });
      }
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: strOf(block.name), args: isRecord(block.input) ? block.input : {} } });
        }
      }
      contents.push({ role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: strOf(raw.tool_call_id),
              response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
            },
          },
        ],
      });
      continue;
    }
  }
  out.contents = contents;

  const declarations = geminiDeclarationsOf(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];

  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response parsing: a provider's answer (JSON or SSE) -> one neutral answer.
// ---------------------------------------------------------------------------

function openAIToolCallsOf(message: Json): UpstreamAnswer['toolCalls'] {
  const out: UpstreamAnswer['toolCalls'] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    out.push({
      id: strOf(call.id),
      name: strOf(call.function.name),
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  return out;
}

/** An OpenAI Chat Completions answer (JSON body or SSE text) as one neutral answer. */
export function parseOpenAIAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  const empty: UpstreamAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] };
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : textJoined(message.content),
      inputTokens: numOrZero(usage.prompt_tokens),
      outputTokens: numOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
      toolCalls: openAIToolCallsOf(message),
    };
  }
  const answer: UpstreamAnswer = { ...empty, toolCalls: [] };
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]' || payload === '') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      answer.inputTokens = numOrZero(chunk.usage.prompt_tokens);
      answer.outputTokens = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answer.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      const slot = byIndex.get(index) ?? { id: '', name: '', arguments: '' };
      if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
      if (isRecord(call.function)) {
        if (typeof call.function.name === 'string' && call.function.name !== '') slot.name = call.function.name;
        if (typeof call.function.arguments === 'string') slot.arguments += call.function.arguments;
      }
      byIndex.set(index, slot);
    }
    if (typeof choice.finish_reason === 'string') answer.stop = choice.finish_reason;
  }
  answer.toolCalls = [...byIndex.values()].filter((c) => c.name !== '');
  return answer;
}

/** An Anthropic Messages answer (JSON body or SSE text) as one neutral answer. */
export function parseAnthropicAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  const empty: UpstreamAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] };
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.content)) return null;
    const blocks: Json[] = value.content.filter(isRecord);
    const usage = isRecord(value.usage) ? value.usage : {};
    const toolCalls: UpstreamAnswer['toolCalls'] = [];
    for (const block of blocks) {
      if (block.type !== 'tool_use') continue;
      toolCalls.push({
        id: strOf(block.id),
        name: strOf(block.name),
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
    return {
      text: textJoined(value.content),
      inputTokens: numOrZero(usage.input_tokens),
      outputTokens: numOrZero(usage.output_tokens),
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : null,
      toolCalls,
    };
  }
  const answer: UpstreamAnswer = { ...empty, toolCalls: [] };
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          answer.inputTokens = numOrZero(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_start': {
        const index = numOrNull(data.index) ?? -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0) {
          byIndex.set(index, { id: strOf(block.id), name: strOf(block.name), arguments: '' });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numOrNull(data.index) ?? -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') answer.text += delta.text;
        if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && index >= 0) {
          const slot = byIndex.get(index);
          if (slot !== undefined) slot.arguments += delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') answer.stop = data.delta.stop_reason;
        if (isRecord(data.usage)) answer.outputTokens = numOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  answer.toolCalls = [...byIndex.values()].filter((c) => c.name !== '');
  return answer;
}

/** A Gemini generateContent answer (JSON body or SSE text) as one neutral answer. */
export function parseGeminiAnswer(text: string, stream: boolean): UpstreamAnswer | null {
  const empty: UpstreamAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] };
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: parts
        .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
        .join(''),
      inputTokens: numOrZero(usage.promptTokenCount),
      outputTokens: numOrZero(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
      toolCalls: [],
    };
  }
  const answer: UpstreamAnswer = { ...empty, toolCalls: [] };
  for (const frame of parseSseFrames(text)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') answer.text += part.text;
    }
    if (typeof candidate.finishReason === 'string') answer.stop = candidate.finishReason;
    if (isRecord(value.usageMetadata)) {
      answer.inputTokens = numOrZero(value.usageMetadata.promptTokenCount);
      answer.outputTokens = numOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return answer;
}

// ---------------------------------------------------------------------------
// Response building: one neutral answer -> the client's wire, JSON or SSE.
// ---------------------------------------------------------------------------

/** One neutral answer as an OpenAI Chat Completions body. */
export function openAIAnswerBody(answer: UpstreamAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_unknown',
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** One neutral answer as an OpenAI Chat Completions SSE stream. */
export function openAIAnswerStream(answer: UpstreamAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One neutral answer as an Anthropic Messages body. */
export function anthropicAnswerBody(answer: UpstreamAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: Json = {};
    if (call.arguments.trim() !== '') {
      try {
        const parsed: unknown = JSON.parse(call.arguments);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
    }
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One neutral answer as an Anthropic Messages SSE stream. */
export function anthropicAnswerStream(answer: UpstreamAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }),
    );
    for (const piece of splitText(answer.text)) {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.arguments },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
