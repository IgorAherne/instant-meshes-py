/*
    options.js: what the embedding page asked of the viewer.

    A host page shapes the viewer with query parameters on its <iframe> URL,
    all optional -- a bare /viewer?session=<id> is the viewer as it always was:

        panel=left|right              the side the control panel docks on; on
                                      the right, the readout moves to the
                                      stage's bottom-right corner with it
        export_label=<text>           the Export button's text ("Accept"...)
        export_action=download|host   download: Export writes the file and the
                                      browser downloads it. host: the button
                                      writes nothing and asks the page instead
        host_origin=<origin>          the page's origin, which host mode needs:
                                      messages go only to it and are taken only
                                      from it

    Host mode is a conversation over window.postMessage, both ways checked
    against host_origin and the parent window (see main.js HostLink, and
    README_PYTHON.md for the messages).  Read here, once, by a pure function,
    so the rules can be tested without a browser.
*/

export const PANEL_SIDES = Object.freeze(['left', 'right']);
export const EXPORT_ACTIONS = Object.freeze(['download', 'host']);
export const DEFAULT_EXPORT_LABEL = 'Export';

/** Longer labels are cut short with an ellipsis: the button is a panel wide
 *  and one line tall. */
export const MAX_EXPORT_LABEL = 32;

/**
 * An origin as the browser writes it ("http://127.0.0.1:7770"), or null.
 *
 * Only http and https, and only a bare origin: a path, a query or a "*" is
 * refused rather than trimmed, because a host that wrote one has not said
 * which page it means, and host mode must never talk to a page it guessed.
 *
 * @param {string|null|undefined} raw
 * @returns {string|null}
 */
export function hostOriginOf(raw) {
    if (!raw) return null;
    let url;
    try {
        url = new URL(raw);
    } catch {
        return null;
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    if (url.username || url.password || url.search || url.hash) return null;
    if (url.pathname !== '/') return null;
    return url.origin;
}

/**
 * The viewer's options, from its URL's query string.
 *
 * Anything missing, unknown or unusable falls back to the default, so a typo
 * gives the viewer as it always was rather than a broken one. Each fallback
 * that the host would want to know about is named in `warnings`, which the
 * viewer writes to the console.
 *
 * @param {string} search    location.search
 * @param {{embedded?: boolean}} [where]  whether the viewer has a parent
 *        page to talk to; host mode is refused without one
 * @returns {{panel: string, exportLabel: string, exportAction: string,
 *            hostOrigin: string|null, warnings: string[]}}
 */
export function readViewerOptions(search, { embedded = true } = {}) {
    const params = new URLSearchParams(search || '');
    const warnings = [];

    const side = params.get('panel');
    let panel = 'left';
    if (side !== null) {
        if (PANEL_SIDES.includes(side)) {
            panel = side;
        } else {
            warnings.push(`panel=${side} is not left or right; the panel stays on the left`);
        }
    }

    const label = (params.get('export_label') || '').trim().replace(/\s+/g, ' ');
    let exportLabel = label || DEFAULT_EXPORT_LABEL;
    if (exportLabel.length > MAX_EXPORT_LABEL) {
        exportLabel = `${exportLabel.slice(0, MAX_EXPORT_LABEL - 1).trimEnd()}…`;
    }

    const wanted = params.get('export_action');
    let exportAction = 'download';
    let hostOrigin = null;
    if (wanted !== null && !EXPORT_ACTIONS.includes(wanted)) {
        warnings.push(`export_action=${wanted} is not download or host; Export downloads`);
    } else if (wanted === 'host') {
        hostOrigin = hostOriginOf(params.get('host_origin'));
        if (!hostOrigin) {
            warnings.push('export_action=host needs host_origin=<the page\'s origin>, ' +
                'e.g. http://127.0.0.1:7770; Export downloads instead');
        } else if (!embedded) {
            warnings.push('export_action=host needs the viewer inside a page (an iframe); ' +
                'Export downloads instead');
            hostOrigin = null;
        } else {
            exportAction = 'host';
        }
    }

    return { panel, exportLabel, exportAction, hostOrigin, warnings };
}
