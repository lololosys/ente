# Wedding link: folders, "find me" and search on a shared album

Date: 2026-10-22. Owner: the Ente account at photos.adeledvir.online (self-hosted, single user).
Fork: github.com/lololosys/ente, branch `feat/album-folders` (web image already deployed as
`ente-web:folders-042edd4107`, built with `web/Dockerfile.photos-overlay`).

## Why

The couple shared one public link to the album "wedding" (1,285 photos, every photo of the day).
Guests open it and see one long list. Guests mostly want to find **the photos they are in**,
then specific moments. The couple also wants the link to show the three parts of the day as
folders, the way their own gallery sidebar now does.

## Goals

1. On the shared wedding link, show three folder tiles above the grid: **Couple : Groupes**,
   **Mairie**, **Repas**. Clicking one shows only that part's photos on the same page; a clear
   "All photos" control returns to everything.
2. **Find me**: a guest takes or uploads a selfie and sees every photo they appear in.
3. **Search**: a guest types words ("first dance", "gâteau", "woman in the green dress",
   "kids at the table") and sees matching photos. French and English must both work (the
   wedding is French-speaking; album names are French).
4. All three narrow the same grid, so the existing viewer, download and share actions keep
   working on the result.

## Non-goals

- No change to the owner's gallery (sidebar folders already shipped), mobile apps or museum.
- No accounts, no guest uploads, no comments work.
- No new general "public album search" product for other albums: it is configured for the
  wedding link. Keep the code generic enough to add a second album by config, nothing more.
- No upstreaming to ente/ente.

## Anti-goals (fail even if it "works")

- Faces or selfies leaving the box. Face detection, embeddings and selfie matching run only on
  our server with Ente's own models. Selfies are never written to disk or logs.
- The OpenRouter key reaching the browser or the repo.
- Breaking the existing link for guests who never touch the new controls, or breaking the
  other shared links (Couple : Groupes, Repas, single-photo links).
- Anyone without the wedding link token being able to query faces or search.

## Decisions already made by the owner

- Build our own modified copy (fork) and run it on the box.
- Photos may be sent (as reduced-size copies) to an AI provider through OpenRouter to write
  descriptions for search. Prefer a provider that does not retain or train on data.
- OpenRouter key is provided; it lives only in a root-only env file on the box. It will be
  rotated after launch (the current one was pasted in chat).
- The account was moved off the `free` plan (public links now render). The wedding album holds
  every photo (added, not moved).

## Constraints and facts

- Box: Hetzner ubuntu-8gb-hel1-2, 4 vCPU, 8 GB RAM, no GPU, 87 GB free. Docker compose in
  `/opt/ente` (services `ente-museum`, `ente-postgres`, `ente-web`), Caddy in `/opt/caddy` on the
  external `web` network. Cloudflare proxies `photos-albums.adeledvir.online` → Caddy →
  `ente-web:3002`. Single-file bind mounts go stale when the host file is replaced
  (`docker exec` copy + `caddy reload` is the recipe).
- Ente is end-to-end encrypted. A public link carries the access token `t` (query) and the
  collection key (URL fragment, base58 if < 50 chars, else hex). With both, any client can list
  and decrypt the album: `GET /public-collection/info`, paged `GET /public-collection/diff`,
  file URLs via `/public-collection/files/{thumbnail|download}/v3/:id`; headers
  `X-Auth-Access-Token` (+ `X-Auth-Link-Device-Token` echo). File key = secretbox open of
  `encryptedKey`/`keyDecryptionNonce` with the collection key; thumbnail = one secretstream blob;
  file = chunked secretstream. Reference: `web/apps/albums/src/public-album/data/api/public-collection.ts`,
  `web/packages/media/file.ts` (`decryptRemoteFile`), `web/packages/base/crypto/libsodium.ts`,
  Rust equivalents in `rust/crates/core/src/crypto`.
- Ente file IDs are global: the same photo has the same ID in "wedding" and in its part album.
  So part membership = the set of file IDs each part album's link lists.
- Ente's own ML models are public ONNX files at `https://models.ente.com/` (`rust/crates/ml/src/assets.rs`):
  YOLOv5-face 640 + MobileFaceNet (aligned 112×112 crops, L2-normalised embeddings), MobileCLIP-S2
  image + quantised text encoder. Thresholds the app uses: same-person cosine ≥ 0.76
  (`web/packages/new/photos/services/ml/cluster.ts:240`), CLIP match dot ≥ 0.175 (`clip.ts:38`).
  `rust/crates/ml` is a complete Rust pipeline; `rust/bindings/napi/photos` exposes it to Node.
- The museum device limit counts `/info` and `/diff` callers per IP + user agent; the indexer
  takes one slot (the owner is no longer on the free plan, so the cap is the paid one).
- The public album page builds its grid from `publicFiles` in
  `web/apps/albums/src/public-album/page/PublicAlbumPage.tsx` (~line 790, trip layout ~737).
  Filtering = a memoised subset of `publicFiles` by file ID.
- Wedding is one day, photos are JPEGs from a photographer (many large group shots: faces can be
  small, so detect on the original or a large preview, not the 720 px thumbnail).

## The design

### Parts

1. **`ente-finder` service** (new, in the fork under `finder/`), one container on the box,
   joined to the `web` network, no public port. Two jobs:
   - **Indexer** (CLI command, run by hand when the album changes; no scheduler):
     for each configured link (wedding + the three part links) list files via the public API,
     decrypt, and for the wedding album:
     - download + decrypt each original (or the largest available image; videos: thumbnail
       only), run YOLOv5-face + MobileFaceNet with Ente's models, store per-face
       `(file_id, box, embedding)`;
     - run MobileCLIP image encoder, store `(file_id, embedding)`;
     - send a reduced copy (long side ~1024 px, JPEG) to an OpenRouter vision model and store
       a short description in **English and French** plus a few tags (people count, setting,
       moment, notable clothing/colours, objects). Model chosen by the builder from cheap
       vision models; a provider with no-retention terms is preferred. Expected cost < $1 for
       1,285 photos. Idempotent: skip files already described.
     - for each part link store the set of file IDs and the part's display name (album name
       with the "Wedding · " prefix removed).
     Storage: one SQLite file in a named volume. Decrypted photos are not kept after
     processing.
   - **API** (small HTTP server) behind Caddy at
     `https://photos-albums.adeledvir.online/finder/*` (same origin as the page, so no CORS):
     - `GET /finder/config` → `{ folders: [{ name, coverFileID, fileIDs[] }], features: { findMe, search } }`
     - `POST /finder/find-me` (multipart selfie, ≤ 8 MB) → `{ fileIDs[], faces: n }`. Detect the
       largest face in the selfie, embed, return wedding files with any face at cosine ≥ a
       threshold starting at 0.6 (selfie vs. candid is looser than the app's 0.76 clustering;
       tune on real photos), ordered by best score. Zero or several faces in the selfie → a
       clear error the UI can show ("we couldn't see a face", "only one person please").
       Selfie bytes stay in memory only.
     - `GET /finder/search?q=` → `{ fileIDs[] }`. Combine: MobileCLIP text score over image
       embeddings (English; translate a non-English query first, see below) and a text match
       over the stored EN/FR descriptions + tags. Rank by a simple blend; cap results.
       Translation of the query to English uses OpenRouter with a small text model (query text
       only, no images), cached.
     - Every endpoint requires the wedding link's access token in `X-Auth-Access-Token` and
       rejects others. If the link is password-protected, also require the link JWT and check
       it with the museum (`/public-collection/info` with that JWT succeeds). Basic per-IP rate
       limit (for example find-me 10/min, search 60/min).
     - Config (env file, root-only): links (token + key) for wedding and the three parts,
       OpenRouter key, model ids, museum URL (internal `http://ente-museum:8080`).
2. **Albums app changes** (`web/apps/albums`), shipped in the same overlay image (extend
   `Dockerfile.photos-overlay` to also rebuild the albums app):
   - On load, if the page's token matches a configured link (`GET /finder/config` returns 200),
     show above the grid: the folder tiles (cover thumbnail from the already-decrypted file,
     name, count), a "Find me" button and a search field. Otherwise render exactly as today.
   - Find me: camera capture on phones (`<input type=file accept=image/* capture=user>`),
     file picker on desktop, short note that the selfie is only used to search and is not
     stored. Shows a result header "Photos of you (N)" with a clear button.
   - Search: debounced submit, result header "N photos for “…”" with clear.
   - Folder, find-me and search each replace the grid's file list with the matching subset
     of `publicFiles` (only one active at a time). Viewer, download, select all work on the
     subset. Back button / Escape clears.
   - Copy in English and French (follow the app's i18n; French strings needed).
3. **Ops**: compose service `ente-finder` in `/opt/ente/compose.yaml`; Caddy `handle_path
   /finder/*` → `ente-finder:<port>` inside the `photos-albums` block, before the existing
   proxy. Create a public link for **Mairie** (it has none; owner approved the plan).

### What "done" looks like

- Opening the wedding link shows the three folders; each shows exactly the photos of that part
  album.
- A selfie of a person known to be in the photos returns most of their photos with few
  strangers (the owner checks a few guests by eye).
- Searches like "gâteau", "cake", "first dance", "mairie", "robe verte" return sensible photos.
- Other shared links look and behave exactly as before.
- No face data, selfie or key appears in logs; the key is only in the env file.

## Owner decision on face search (2026-10-22)

Option 2: build find-me **with an exclusion list**. The owner can exclude a person (a guest who
asks, or the children) so that photos are never returned for selfies matching that person:
- the indexer clusters faces (Ente's 0.76 rule) and the owner-only CLI lists clusters with a
  sample crop path for the owner to view on the box, then `exclude <cluster>` adds every face in
  that cluster to an exclusion list; find-me never matches excluded faces;
- the exclusion list is checked at query time, so it applies immediately without re-indexing;
- a `forget-faces` command deletes all face data (embeddings, clusters) in one step.

## Calls made by Claude (owner can override)

- **Result order.** Folders show their photos in the normal date-grouped grid. Find-me and
  search show a flat grid best match first (no date grouping); viewer next/previous follow that
  order. Weak matches are cut by threshold and a cap.
- **Tuning find-me without guest selfies.** Use face crops from the album itself as stand-in
  selfies, check matches by eye, set the threshold from that. An owner-only CLI prints per-file
  scores; nothing face-related is logged. The couple tries it with their own selfies after launch.
- **Mairie link** is created from the owner's already signed-in browser session. No password or
  device limit added to any link.
- **Rollout.** Direct deploy, previous web image tag held for a one-command rollback. New image
  first started on a spare port and checked against the wedding link and one other shared link.
  If the finder API is down the page renders exactly as today (controls hidden).

## Left to the builder

- Language for `ente-finder` (Python + onnxruntime + PyNaCl, or Node/Rust reusing
  `rust/crates/ml` via the napi binding). Pick whichever reproduces Ente's preprocessing most
  faithfully with least code; verify embeddings against a known pair of photos.
- Exact OpenRouter models, prompt, score blend, result caps, thresholds (tune on this album).
- UI layout details, matching the albums app look (it already has trip/masonry layouts).
- Tests: unit tests for crypto helpers (round-trip against a fixture made with the web crypto
  code), the filter logic, and the API auth; a manual end-to-end check on the live link.
