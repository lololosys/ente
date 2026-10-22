import type { EnteFile } from "ente-media/file";

/**
 * Our self-hosted "finder" service (see finder/ at the repo root) answers for
 * one shared album at the same origin, under /finder/. Every call carries the
 * link's access token; any failure means the page renders as it always did.
 */
const FINDER_BASE = "/finder";

export interface FinderFolder {
    name: string;
    coverFileID: number;
    fileIDs: number[];
}

export interface FinderConfig {
    folders: FinderFolder[];
    features: { findMe: boolean; search: boolean };
}

export type FinderFilter =
    | { kind: "folder"; name: string; fileIDs: number[] }
    | { kind: "find-me" | "search"; label: string; fileIDs: number[] };

export type FindMeError = "no_face" | "many_faces" | "slow_down" | "failed";

const headers = (accessToken: string) => ({
    "X-Auth-Access-Token": accessToken,
});

export const fetchFinderConfig = async (
    accessToken: string,
): Promise<FinderConfig | undefined> => {
    try {
        const res = await fetch(`${FINDER_BASE}/config`, {
            headers: headers(accessToken),
        });
        if (!res.ok) return undefined;
        return (await res.json()) as FinderConfig;
    } catch {
        return undefined;
    }
};

export const findMe = async (
    accessToken: string,
    selfie: File,
): Promise<{ fileIDs: number[] } | { error: FindMeError }> => {
    const body = new FormData();
    body.append("selfie", selfie);
    try {
        const res = await fetch(`${FINDER_BASE}/find-me`, {
            method: "POST",
            headers: headers(accessToken),
            body,
        });
        if (res.status == 429) return { error: "slow_down" };
        const json = (await res.json()) as {
            fileIDs?: number[];
            error?: string;
        };
        if (res.ok && json.fileIDs) return { fileIDs: json.fileIDs };
        return {
            error:
                json.error == "no_face" || json.error == "many_faces"
                    ? json.error
                    : "failed",
        };
    } catch {
        return { error: "failed" };
    }
};

export const searchFinder = async (
    accessToken: string,
    q: string,
): Promise<number[] | undefined> => {
    try {
        const res = await fetch(
            `${FINDER_BASE}/search?${new URLSearchParams({ q })}`,
            { headers: headers(accessToken) },
        );
        if (!res.ok) return undefined;
        return ((await res.json()) as { fileIDs: number[] }).fileIDs;
    } catch {
        return undefined;
    }
};

/**
 * Narrows the album's files to a filter's IDs. Folders keep the album's own
 * (date) order; find-me and search keep the service's best-match-first order.
 */
export const applyFinderFilter = (
    files: EnteFile[],
    active: FinderFilter | undefined,
): { files: EnteFile[]; ranked: boolean } => {
    if (!active) return { files, ranked: false };
    if (active.kind == "folder") {
        const ids = new Set(active.fileIDs);
        return { files: files.filter((f) => ids.has(f.id)), ranked: false };
    }
    const byID = new Map(files.map((f) => [f.id, f]));
    return {
        files: active.fileIDs.flatMap((id) => {
            const f = byID.get(id);
            return f ? [f] : [];
        }),
        ranked: true,
    };
};
