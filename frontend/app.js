/* Restaurant Analysis — video client (vanilla JS, no framework).
 *
 * Talks to one endpoint only:
 *   GET /stream  — the processed MJPEG video published by the background
 *                  processing thread (detection and on-video dashboard
 *                  included in the frames themselves)
 * Nothing is polled and no statistics are requested: the page shows the
 * frames exactly as the backend renders them.
 */

"use strict";

(function () {
  const RECONNECT_MS = 3000;
  const FRAME_CHECK_MS = 500;

  const stream = document.getElementById("stream");
  const overlay = document.getElementById("video-overlay");
  const overlayText = document.getElementById("video-overlay-text");

  let reconnectTimer = null;

  function connectStream() {
    // The cache-buster makes the browser open a fresh connection to /stream.
    stream.src = `/stream?t=${Date.now()}`;
  }

  // Hide the connecting overlay once the first frame has decoded. This only
  // reads the image's natural size; it makes no requests of its own.
  setInterval(() => {
    if (stream.naturalWidth > 0) {
      overlay.hidden = true;
    }
  }, FRAME_CHECK_MS);

  stream.addEventListener("error", () => {
    if (reconnectTimer !== null) {
      return; // a reconnect is already scheduled
    }
    overlay.hidden = false;
    overlayText.textContent = "Processed video unavailable — retrying…";
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectStream();
    }, RECONNECT_MS);
  });

  connectStream();
})();
