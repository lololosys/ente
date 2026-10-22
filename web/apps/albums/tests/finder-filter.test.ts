import type { EnteFile } from "ente-media/file";
import { afterEach, describe, expect, test, vi } from "vitest";
import {
    applyFinderFilter,
    fetchFinderConfig,
    findMe,
} from "../src/public-album/finder/filter";

const file = (id: number) => ({ id }) as unknown as EnteFile;
const files = [file(1), file(2), file(3), file(4)];

describe("finder filter", () => {
    test("no active filter returns files unchanged", () => {
        expect(applyFinderFilter(files, undefined)).toEqual({
            files,
            ranked: false,
        });
    });

    test("folder filter keeps date order and drops unknown ids", () => {
        const r = applyFinderFilter(files, {
            kind: "folder",
            name: "Repas",
            fileIDs: [4, 2, 99],
        });
        expect(r.files.map((f) => f.id)).toEqual([2, 4]);
        expect(r.ranked).toBe(false);
    });

    test("find-me filter keeps API order and is ranked", () => {
        const r = applyFinderFilter(files, {
            kind: "find-me",
            label: "",
            fileIDs: [3, 1, 99],
        });
        expect(r.files.map((f) => f.id)).toEqual([3, 1]);
        expect(r.ranked).toBe(true);
    });
});

describe("fetchFinderConfig", () => {
    afterEach(() => vi.unstubAllGlobals());

    test("returns undefined on 401", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.resolve(new Response("{}", { status: 401 }))),
        );
        expect(await fetchFinderConfig("T")).toBeUndefined();
    });

    test("returns undefined on network error", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.reject(new TypeError("offline"))),
        );
        expect(await fetchFinderConfig("T")).toBeUndefined();
    });

    test("sends the access token and parses the config", async () => {
        const fetchMock = vi.fn((...args: [string, RequestInit?]) =>
            Promise.resolve(
                new Response(
                    JSON.stringify({
                        folders: [
                            { name: "Repas", coverFileID: 2, fileIDs: [2] },
                        ],
                        features: { findMe: true, search: true },
                    }),
                    { headers: { "x-args": String(args.length) } },
                ),
            ),
        );
        vi.stubGlobal("fetch", fetchMock);
        const config = await fetchFinderConfig("T");
        expect(config?.folders[0]?.name).toBe("Repas");
        expect(fetchMock.mock.calls[0]![0]).toBe("/finder/config");
        expect(
            (fetchMock.mock.calls[0]![1]!.headers as Record<string, string>)[
                "X-Auth-Access-Token"
            ],
        ).toBe("T");
    });
});

describe("findMe", () => {
    afterEach(() => vi.unstubAllGlobals());

    test("a selfie that is too large gets its own error", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() =>
                Promise.resolve(
                    new Response(JSON.stringify({ error: "too_large" }), {
                        status: 413,
                    }),
                ),
            ),
        );
        expect(await findMe("T", new File(["x"], "s.jpg"))).toEqual({
            error: "too_large",
        });
    });
});
