# VisionGuard Home / Client Update — 2.1.0

## Dashboard camera fixes
- Added administrator-only camera source read/edit flow. RTSP credentials are returned only to administrators during edit; normal camera listings continue to redact them.
- Added a Test connection action that restarts the camera worker and returns live telemetry.
- Camera source changes now restart the worker. Camera configuration updates merge with stored settings rather than replacing the full configuration.
- The MJPEG stream now returns a reconnect/status placeholder when a previously active camera drops, rather than leaving the last frame frozen.
- Worker frame buffers are cleared while reconnecting.

## Admin and client workspaces
- Admin creates separate client usernames/passwords and assigns cameras to each client.
- Client passwords are salted PBKDF2-HMAC-SHA256 hashes.
- Clients see only their assigned cameras, event history, statistics, snapshots, and streams.
- Clients can change detection configuration and zones for their assigned cameras, but cannot add/delete cameras, change stream URLs, reassign cameras, or manage client accounts.
- Client accounts can be disabled by an administrator.
- Admin login is configured using ADMIN_USERNAME and ADMIN_PASSWORD. SESSION_SECRET signs session cookies. Set both to unique values before creating clients.

## Alerts and time
- Removed Telegram configuration controls and disabled Telegram sending in this home/client build. No bot token or chat ID is required.
- Offline TTS alerts remain available.
- Dashboard clock and event display use Asia/Kolkata (IST); event records remain stored in UTC.

## Run and upgrade
1. Stop the running server and back up the existing `data/visionguard.db` and `.env`.
2. Extract this release over a clean project folder (do not overwrite your `.env`, `data`, or `models` folders).
3. Add ADMIN_USERNAME, ADMIN_PASSWORD, SESSION_SECRET, and APP_TIMEZONE=Asia/Kolkata to `.env`.
4. Run `install_windows.bat` only if dependencies are not already installed; otherwise run `run.bat`.
5. Sign in as admin, create client workspaces, add cameras, assign each camera, and use Test connection.

## Validation
- Python compileall: passed.
- JavaScript syntax check: passed.
- Pytest: 11 passed.
- Physical camera/RTSP streaming was not testable in this build environment; verify each camera on the target Windows PC.


## Camera loading and confidence update
- Moved initial USB camera discovery until after SQLite initialization.
- Added administrator-only USB camera discovery and source preflight endpoints.
- Added dashboard buttons to discover USB cameras and test a source before saving. Offline sources can still be saved and appear with worker connection errors/status.
- Dashboard camera loading now displays a Retry/error state instead of remaining on “Loading cameras…”. Dashboard sections load independently so a stats/events failure cannot block the camera list.
- Added duplicate source protection.
- Windows webcam capture prefers DirectShow, then falls back to OpenCV default.
- Model confidence thresholds are clamped to >= 0.50; lower-confidence model events are discarded before any event or alert is created.
- Telegram is disabled; no Telegram bot credentials are required.


## Telegram suspicious-activity alerts

- Telegram is optional and requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
- Per-camera `Telegram Security Alerts` switch controls delivery.
- Only allow-listed suspicious/security event types are eligible.
- Confidence must be strictly greater than 75% (default `TELEGRAM_MIN_CONFIDENCE=0.76`). A score of exactly 75% is not sent.
- Events without an explicit model confidence score (for example, pure line-crossing analytics) are not sent.
- Notifications are rate-limited per camera and event type.
- Existing installations get a one-time migration to enable the per-camera switch; user changes are preserved on later restarts.
