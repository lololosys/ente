# Wedding Link Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the shared wedding link, add folder tiles, selfie "Find me" and word search, served by a small indexer + API on the photo box.

**Architecture:** A Python service `finder/` (CLI indexer + HTTP API) decrypts the public albums with their link token + key, runs Ente's own ONNX face and MobileCLIP models on CPU, gets EN/FR captions from OpenRouter, and stores everything in SQLite. The albums web app calls it same-origin under `/finder/*` (Caddy route) and filters its existing grid. Faces never leave the box.

**Tech Stack:** Python 3.12, onnxruntime (CPU), PyNaCl (secretbox + secretstream), numpy, opencv-python-headless, Pillow, regex, httpx, starlette + uvicorn, sqlite3, base58; pytest. Web: Next.js albums app (TypeScript, MUI), vitest.

**Spec:** `docs/superpowers/specs/2026-10-22-wedding-link-finder-design.md`

## Global Constraints

- Selfies: in memory only; never written to disk, never logged. No face data or OpenRouter key in logs.
- OpenRouter key and link keys only in `/opt/ente-finder/finder.env` (mode 600, root). Never in the repo or the browser.
- Every `/finder/*` call requires `X-Auth-Access-Token` equal to the wedding link token, else 401.
- Finder down or 401 → albums page renders exactly as today (no controls).
- Same-person rule for clustering: cosine ≥ 0.76. Find-me starting threshold: cosine ≥ 0.60 (tuned in Task 7). CLIP match floor: dot ≥ 0.175.
- Rate limits per IP: find-me 10/min, search 60/min.
- Selfie upload ≤ 8 MB. Caption image long side ≈ 1024 px JPEG.
- Ente model URLs: `https://models.ente.com/yolov5s_face_640_640_static_b1.onnx`, `mobilefacenet_portable_static_b1.onnx`, `mobileclip_s2_image_gelu_opset20.onnx`, `mobileclip_s2_text_opset18_quant.onnx`, `bpe_simple_vocab_16e6.txt`; verify sha256 against `rust/crates/ml/src/assets.rs`.
- Preprocessing mirrors `rust/crates/ml/src/preprocess.rs` (YOLO: 640 letterbox, pad 114, /255; face: 5-landmark similarity align to 112×112 with `MOBILEFACENET_IDEAL_5_LANDMARKS`, (x-127.5)/127.5, L2-normalise; CLIP: 256 resize + centre crop) and `face/detect.rs` (IoU 0.4, min score 0.5).
- No scheduler: indexing is a CLI run by hand.
- UI copy in English and French.

## Review Focus

1. Selfie with no face / two faces / a photo of a screen → clear message, no 500 (Task 4 test).
2. Video and HEIC/RAW files in the album → indexer skips the original, uses the thumbnail, never crashes the run (Task 2 test).
3. Wedding link token rotated or link deleted by owner → API returns 401, page shows no controls (Task 5 + Task 6 tests).
4. Excluded cluster → its faces never returned even when the selfie matches strongly (Task 4 test).
5. Indexer re-run after new photos added → only new files processed, captions not re-bought (Task 3 test).

---

### Task 1: Link crypto + public API client

**Files:**
- Create: `finder/pyproject.toml`, `finder/finder/__init__.py`, `finder/finder/crypto.py`, `finder/finder/ente_api.py`
- Test: `finder/tests/test_crypto.py`, `finder/tests/test_ente_api.py`

**Interfaces:**
- Produces: `collection_key_from_fragment(fragment: str) -> bytes` (base58 if len < 50 else hex); `open_box(cipher_b64: str, nonce_b64: str, key: bytes) -> bytes`; `decrypt_blob(data: bytes, header_b64: str, key: bytes) -> bytes` (single secretstream message); `decrypt_stream(data: bytes, header_b64: str, key: bytes) -> bytes` (chunks of 4 MiB + ABYTES); `PublicAlbum(museum_url: str, token: str, key: bytes)` with `.info() -> dict`, `.files() -> list[RemoteFile]` (paged `/public-collection/diff`, drops `isDeleted`), `.thumbnail(f) -> bytes`, `.original(f) -> bytes`; `RemoteFile` dataclass: `id, key: bytes, file_type: int, title: str, thumb_header, file_header`.

- [ ] **Step 1: Write failing tests.** `test_fragment_base58_and_hex` (round-trip of a 32-byte key both ways). `test_open_box_round_trip`, `test_blob_round_trip`, `test_stream_round_trip_multi_chunk` (encrypt with PyNaCl bindings, 9 MiB payload). `test_files_pages_until_has_more_false` with an `httpx.MockTransport` returning two diff pages and one deleted entry → 2 files, header `X-Auth-Access-Token` sent, `X-Link-Device-Token` echoed on the second call as `X-Auth-Link-Device-Token`.
- [ ] **Step 2:** `cd finder && pytest -q` → FAIL (module missing).
- [ ] **Step 3:** Implement per `web/apps/albums/src/public-album/data/api/public-collection.ts`, `web/packages/media/file.ts::decryptRemoteFile`, `web/packages/base/crypto/libsodium.ts`. Download URLs via `/public-collection/files/{thumbnail|download}/v3/:id` (returns signed URL), fallback to `/public-collection/files/{preview|download}/:id`.
- [ ] **Step 4:** `pytest -q` → PASS. Also a live smoke (not committed): list the wedding link → 1,285 files, decrypt one thumbnail to a valid JPEG.
- [ ] **Step 5:** Commit `finder: link crypto and public album client`.

### Task 2: Store + indexer skeleton + folders

**Files:**
- Create: `finder/finder/store.py`, `finder/finder/config.py`, `finder/finder/images.py`, `finder/finder/cli.py`
- Test: `finder/tests/test_store.py`, `finder/tests/test_indexer.py`

**Interfaces:**
- Consumes: Task 1 `PublicAlbum`, `RemoteFile`.
- Produces: `Config.from_env()` with `museum_url, main: LinkRef, parts: list[LinkRef], openrouter_key, caption_model, translate_model, db_path, models_dir`; `LinkRef(name: str, token: str, key: bytes)` parsed from env `FINDER_MAIN_LINK` / `FINDER_PART_LINKS` (full share URLs, `;`-separated, name = album name minus text up to " · "). `Store(path)` tables: `files(id PK, done_faces, done_clip, done_caption)`, `faces(id PK, file_id, x1,y1,x2,y2, score, emb BLOB float32, cluster INT)`, `clip(file_id PK, emb BLOB)`, `captions(file_id PK, en, fr, tags)`, `folders(name, file_id)`, `excluded_clusters(cluster PK)`. `load_image(f: RemoteFile, album: PublicAlbum) -> PIL.Image | None`: images → original (decode with Pillow, EXIF-transpose; HEIC/RAW/undecodable → thumbnail); videos → thumbnail. CLI `python -m finder index [--only faces|clip|captions|folders]`.

- [ ] **Step 1: Write failing tests.** `test_folders_written_from_part_links` (fake albums: part A lists ids {1,2}, part B {3}) → `store.folders()` == `{"A":[1,2],"B":[3]}`. `test_video_uses_thumbnail` and `test_undecodable_original_falls_back_to_thumbnail`. `test_rerun_skips_done_files` (second run calls stage function only for new ids).
- [ ] **Step 2:** `pytest -q` → FAIL.
- [ ] **Step 3:** Implement. Stages are callables registered by later tasks; this task wires the loop and the folders stage.
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5:** Commit `finder: store, config and indexer loop with folders`.

### Task 3: Faces, CLIP and captions stages

**Files:**
- Create: `finder/finder/models.py` (download + sha256 + onnxruntime sessions), `finder/finder/faces.py`, `finder/finder/clip.py`, `finder/finder/clip_tokenizer.py`, `finder/finder/captions.py`
- Test: `finder/tests/test_faces.py`, `finder/tests/test_clip_tokenizer.py`, `finder/tests/test_captions.py`, fixtures in `finder/tests/fixtures/` (2 public-domain portraits of the same person + 1 of another; record source URLs in `fixtures/README.md`)

**Interfaces:**
- Produces: `detect_faces(img) -> list[Face]` (`Face(box, score, landmarks5)`), `embed_face(img, face) -> np.ndarray[float32]` (L2-normalised), `cluster_faces(store)` (greedy, cosine ≥ 0.76), `clip_image(img) -> np.ndarray`, `clip_text(text: str) -> np.ndarray`, `tokenize(text) -> np.ndarray[int64, 77]`, `caption(img, client) -> Caption(en: str, fr: str, tags: list[str])`, `translate_to_english(q: str, client) -> str` (LRU-cached).

- [ ] **Step 1: Write failing tests.** `test_same_person_scores_above_0_6_and_other_below` (fixtures). `test_detect_finds_one_face_in_portrait`. `test_tokenizer_matches_rust_vectors` (copy 3 expected token arrays from `rust/crates/ml/src/clip/tokenizer.rs` tests). `test_clip_text_image_dot_ranks_matching_caption_higher` ("a photo of a dog" vs a dog fixture beats "a photo of a cake"). `test_caption_parses_json_reply` (OpenRouter mocked; prompt asks for JSON `{en, fr, tags}`; malformed reply → one retry then skip, file stays not-done). `test_rerun_does_not_recaption` (done_caption set → no API call).
- [ ] **Step 2:** `pytest -q` → FAIL.
- [ ] **Step 3:** Implement mirroring the Rust preprocessing named in Global Constraints. Caption model default `anthropic/claude-haiku-5.5` (cheap vision, chosen by builder; override via env); request `provider: {data_collection: "deny"}`. Image sent as base64 JPEG long side 1024.
- [ ] **Step 4:** `pytest -q` → PASS. Live: `python -m finder index --only faces --limit 20` on the box, print face counts (no embeddings).
- [ ] **Step 5:** Commit `finder: face, CLIP and caption stages`.

### Task 4: Find-me, search and the owner CLI

**Files:**
- Create: `finder/finder/query.py`
- Modify: `finder/finder/cli.py` (commands `clusters`, `exclude <id>`, `include <id>`, `forget-faces`, `try-face <file_id> <face_index>`)
- Test: `finder/tests/test_query.py`

**Interfaces:**
- Consumes: Task 3 functions, Task 2 `Store`.
- Produces: `find_me(selfie_bytes: bytes) -> FindResult(file_ids: list[int], faces: int)` raising `NoFace` / `ManyFaces`; `search(q: str) -> list[int]`; `folders() -> list[Folder(name, cover_file_id, file_ids)]`.

- [ ] **Step 1: Write failing tests.** `test_find_me_orders_by_best_score_and_cuts_at_threshold`. `test_find_me_skips_excluded_clusters` (score 0.95 but cluster excluded → absent). `test_find_me_no_face_raises_NoFace`, `test_find_me_two_large_faces_raises_ManyFaces` (second face area ≥ 50% of largest), small background faces allowed. `test_search_blends_clip_and_caption` (caption hit "gâteau" in fr ranks first; CLIP-only hit still returned above 0.175). `test_forget_faces_empties_faces_and_clusters`. `clusters` prints cluster id, size and writes one sample crop per cluster to `/var/lib/finder/crops/` (owner views on the box; directory removed by `forget-faces`).
- [ ] **Step 2:** `pytest -q` → FAIL.
- [ ] **Step 3:** Implement. Search: translate non-ASCII-or-French query to English for CLIP; caption match is case/accent-insensitive token overlap over en+fr+tags; score = 0.6·clip_norm + 0.4·caption; cap 300. Find-me cap 500.
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5:** Commit `finder: find-me, search and owner commands`.

### Task 5: HTTP API

**Files:**
- Create: `finder/finder/api.py`, `finder/Dockerfile`
- Test: `finder/tests/test_api.py`

**Interfaces:**
- Consumes: Task 4 `find_me`, `search`, `folders`.
- Produces (paths as seen by the app after Caddy strips `/finder`): `GET /config` → `{folders:[{name,coverFileID,fileIDs}], features:{findMe:bool, search:bool}}`; `POST /find-me` multipart field `selfie` → `{fileIDs, faces}` or 422 `{error:"no_face"|"many_faces"}`; `GET /search?q=` → `{fileIDs}`; 401 on wrong/missing token; 413 over 8 MB; 429 over rate.

- [ ] **Step 1: Write failing tests** (starlette TestClient): `test_401_without_token`, `test_401_wrong_token`, `test_config_shape`, `test_find_me_422_no_face`, `test_413_large_selfie`, `test_rate_limit_find_me_11th_is_429`, `test_selfie_not_written` (patch `open`/tempfile to assert no write), `test_logs_have_no_selfie_or_key` (caplog).
- [ ] **Step 2:** `pytest -q` → FAIL.
- [ ] **Step 3:** Implement; models loaded once at startup; `Dockerfile` python:3.12-slim, non-root user, `uvicorn finder.api:app --port 8090`, models + db in volume `/var/lib/finder`.
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5:** Commit `finder: HTTP API and container`.

### Task 6: Albums app: folders, Find me, search

**Files:**
- Create: `web/apps/albums/src/public-album/finder/finder-api.ts`, `web/apps/albums/src/public-album/finder/FinderBar.tsx`, `web/apps/albums/src/public-album/finder/filter.ts`
- Modify: `web/apps/albums/src/public-album/page/PublicAlbumPage.tsx` (~151 state, ~697 `disableGrouping`, ~790 `files=`), i18n strings (EN + FR) in the albums locale files
- Modify: `web/Dockerfile.photos-overlay` (also `npm run build:albums`, copy to `/out/albums`)
- Test: `web/apps/albums/tests/finder-filter.test.ts`

**Interfaces:**
- Consumes: Task 5 API.
- Produces: `fetchFinderConfig(token: string): Promise<FinderConfig | undefined>` (undefined on any error/401); `findMe(token, file: File)`, `searchFinder(token, q)`; `applyFinderFilter(files: EnteFile[], active: FinderFilter | undefined): { files: EnteFile[]; ranked: boolean }` where `FinderFilter = {kind:"folder", name, fileIDs} | {kind:"find-me"|"search", label, fileIDs}`; ranked results keep API order, folders keep date order.

- [ ] **Step 1: Write failing tests:** `folder filter keeps date order and drops unknown ids`; `find-me filter keeps API order and is ranked`; `no active filter returns files unchanged`; `fetchFinderConfig returns undefined on 401 and on network error`.
- [ ] **Step 2:** `npx vitest --run tests/finder-filter.test.ts` → FAIL.
- [ ] **Step 3:** Implement. `FinderBar` above the grid (only when config loaded and layout ≠ trip): folder tiles (cover from the matching `publicFiles` thumbnail, name, count), "Find me / Me trouver" (`<input type=file accept="image/*" capture="user">`, note "Your selfie is only used for this search and isn't stored" / "Votre selfie sert uniquement à cette recherche et n'est pas conservé"), search field. Active filter shows a chip with count + clear; Escape clears. `files={filtered.files}` and `disableGrouping={disableGrouping || filtered.ranked}`.
- [ ] **Step 4:** vitest PASS; `npx tsc --noEmit -p web/apps/albums` clean except the known wasm pkg errors; eslint + prettier clean on touched files.
- [ ] **Step 5:** Commit `[web][albums] Finder bar on public albums: folders, find me, search`.

### Task 7: Deploy, index, tune

**Files:**
- Modify (box): `/opt/ente/compose.yaml` (service `ente-finder`, env_file `/opt/ente-finder/finder.env`, volume `ente-finder-data:/var/lib/finder`, networks `[ente-internal, web]`), `/opt/caddy/Caddyfile` (`handle_path /finder/*` → `ente-finder:8090` inside `photos-albums.adeledvir.online`, before the existing proxy; apply via the docker-exec copy + `caddy reload` recipe)
- Create (box): `/opt/ente-finder/finder.env` mode 600

- [ ] **Step 1:** Create the Mairie public link in the owner's signed-in browser; collect the four share URLs (wedding + 3 parts) into `finder.env` without echoing them.
- [ ] **Step 2:** Build `ente-finder` and the new web overlay image on the box; tag current web image as rollback. Run new web image on spare port 13002 and check the wedding link and the Repas link render; then switch.
- [ ] **Step 3:** `python -m finder index` (faces, clip, captions, folders). Expected: 1,285 files done; OpenRouter spend < $1 (check `/api/v1/key` usage).
- [ ] **Step 4:** Tune: `try-face` on 5 recurring people; pick the threshold that returns their photos with ≤ ~5% strangers by eye; set `FINDER_FACE_THRESHOLD`.
- [ ] **Step 5:** Browser check on the live link: folders (counts 530 / 354 / 510), find-me with a face crop image, search "gâteau" and "first dance", clear; another shared link unchanged; finder stopped → page as before. Commit any fixes; push branch.
