/*
    net.js: the WebSocket transport for the viewer.

    Frames are the binary ones defined in protocol.js.  The connection survives
    a server restart on its own: it reconnects with capped exponential backoff,
    replays the frames that were produced while it was down, and lets the caller
    re-run its handshake before that replay so the server always sees HELLO
    before it sees a stroke.

    A frame that fails to decode is not retried.  It means the two ends disagree
    about the wire format -- a version bump, most likely -- and reconnecting
    would only repeat the failure, so the connection is reported as fatal and
    left closed for the page to surface.
*/

import { MessageType, ProtocolError, decode, encode } from './protocol.js';

const RECONNECT_BASE_MS = 500;
const RECONNECT_CAP_MS = 15000;
const RECONNECT_JITTER = 0.25;
const MAX_QUEUED_FRAMES = 64;

/** @typedef {'connecting'|'open'|'closed'} ConnectionState */

export class Connection {
    /**
     * @param {string} url
     * @param {{onOpen?: (connection: Connection) => void, maxQueue?: number}} options
     *        onOpen runs while the socket is open but before queued frames are
     *        flushed, which is what makes a handshake arrive first.
     */
    constructor(url, { onOpen = null, maxQueue = MAX_QUEUED_FRAMES } = {}) {
        this.url = url;

        this._onOpen = onOpen;
        this._maxQueue = maxQueue;
        this._handlers = new Map();
        this._statusHandlers = new Set();

        this._socket = null;
        this._state = 'closed';
        this._attempt = 0;
        this._queue = [];
        this._retryTimer = null;
        this._closedByUs = false;
        this._fatal = false;
    }

    /** @returns {ConnectionState} */
    get status() {
        return this._state;
    }

    /** Subscribe to one message type. @returns {() => void} unsubscribe */
    on(type, handler) {
        let set = this._handlers.get(type);
        if (!set) {
            set = new Set();
            this._handlers.set(type, set);
        }
        set.add(handler);
        return () => set.delete(handler);
    }

    /**
     * Subscribe to transport state changes.
     * @param {(status: {state: ConnectionState, attempt: number, url: string}) => void} handler
     * @returns {() => void} unsubscribe
     */
    onStatus(handler) {
        this._statusHandlers.add(handler);
        handler(this._statusSnapshot());
        return () => this._statusHandlers.delete(handler);
    }

    connect() {
        if (this._fatal || this._socket) return;

        this._closedByUs = false;
        clearTimeout(this._retryTimer);
        this._retryTimer = null;
        this._setState('connecting');

        const socket = new WebSocket(this.url);
        socket.binaryType = 'arraybuffer';
        socket.onopen = () => this._handleOpen(socket);
        socket.onmessage = (event) => this._handleMessage(event);
        socket.onerror = () => {
            /* onerror carries no usable detail in browsers; onclose follows and
               is where the retry is scheduled. */
        };
        socket.onclose = () => this._handleClose(socket);
        this._socket = socket;
    }

    /**
     * Send a frame, queueing it when the socket is down.
     *
     * @param {number} type    a MessageType value
     * @param {object} header  JSON-serialisable metadata
     * @param {object|null} arrays  name -> {data: TypedArray, shape: number[]}
     */
    send(type, header = {}, arrays = null) {
        let frame;
        try {
            frame = encode(type, header, arrays);
        } catch (err) {
            this._reportLocalError(`could not encode message ${type}: ${err.message}`, false);
            return;
        }

        if (this._socket && this._socket.readyState === WebSocket.OPEN) {
            this._socket.send(frame);
            return;
        }

        if (this._queue.length >= this._maxQueue) this._queue.shift();
        this._queue.push(frame);
    }

    /** Close for good; no reconnect follows. */
    close() {
        this._closedByUs = true;
        clearTimeout(this._retryTimer);
        this._retryTimer = null;
        this._queue.length = 0;
        if (this._socket) this._socket.close();
        this._socket = null;
        this._setState('closed');
    }

    /* -------------------------------------------------------------- */
    /*  Socket events                                                  */
    /* -------------------------------------------------------------- */

    _handleOpen(socket) {
        if (socket !== this._socket) return;
        this._attempt = 0;
        this._setState('open');

        if (this._onOpen) this._onOpen(this);

        const pending = this._queue;
        this._queue = [];
        for (const frame of pending) socket.send(frame);
    }

    _handleMessage(event) {
        if (!(event.data instanceof ArrayBuffer)) {
            this._reportLocalError('received a text frame; this protocol is binary only', true);
            this._failFatally();
            return;
        }

        let message;
        try {
            message = decode(event.data);
        } catch (err) {
            const fatal = err instanceof ProtocolError;
            this._reportLocalError(`malformed frame: ${err.message}`, fatal);
            if (fatal) this._failFatally();
            return;
        }

        this._dispatch(message.type, message.header, message.arrays);
    }

    _handleClose(socket) {
        if (socket !== this._socket) return;
        this._socket = null;
        this._setState('closed');
        if (this._closedByUs || this._fatal) return;

        this._attempt += 1;
        this._retryTimer = setTimeout(() => this.connect(), backoffDelay(this._attempt));
    }

    /* -------------------------------------------------------------- */
    /*  Internals                                                      */
    /* -------------------------------------------------------------- */

    _dispatch(type, header, arrays) {
        const handlers = this._handlers.get(type);
        if (!handlers) return;
        for (const handler of handlers) handler(header, arrays);
    }

    /**
     * Surface a transport-level failure through the same channel the server
     * uses, so the page needs only one place to render an error.
     */
    _reportLocalError(message, fatal) {
        this._dispatch(MessageType.ERROR, { message, fatal, local: true }, {});
    }

    _failFatally() {
        this._fatal = true;
        this.close();
    }

    _statusSnapshot() {
        return { state: this._state, attempt: this._attempt, url: this.url };
    }

    _setState(state) {
        if (this._state === state) return;
        this._state = state;
        const snapshot = this._statusSnapshot();
        for (const handler of this._statusHandlers) handler(snapshot);
    }
}

/** Exponential backoff, capped, with jitter so reconnects do not synchronise. */
function backoffDelay(attempt) {
    const base = Math.min(RECONNECT_CAP_MS, RECONNECT_BASE_MS * 2 ** (attempt - 1));
    return Math.round(base * (1 + (Math.random() * 2 - 1) * RECONNECT_JITTER));
}
